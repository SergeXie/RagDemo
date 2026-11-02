# ============================================================
# RagDemo 一键启动脚本
# 用法：powershell -ExecutionPolicy Bypass -File .\start.ps1
# 流程：Redis(可选) -> 检查 .env -> 检查依赖 -> 启动后端(8000) + 前端(5173)
# ============================================================

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
$psExe = (Get-Process -Id $PID).Path

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  [提示] $msg" -ForegroundColor Yellow }
function Err($msg)  { Write-Host "  [错误] $msg" -ForegroundColor Red }

# 等待 HTTP 服务就绪（最多 $timeout 秒，失败不中断）
function Wait-Http($url, $timeout, $name) {
    for ($i = 0; $i -lt $timeout * 2; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 400) {
                Ok "$name 就绪：$url"
                return $true
            }
        } catch { Start-Sleep -Milliseconds 500 }
    }
    Warn "$name 暂未响应（$url），请查看对应窗口的输出"
    return $false
}

# ---- 1. Redis（可选，没有 Docker 时自动降级为内存记忆）----
Step "启动 Redis"
if (Get-Command docker -ErrorAction SilentlyContinue) {
    Push-Location $root
    docker compose up -d redis 2>$null
    if ($LASTEXITCODE -ne 0) {
        docker-compose up -d redis 2>$null
    }
    $redisOk = ($LASTEXITCODE -eq 0)
    Pop-Location
    if ($redisOk) { Ok "Redis 容器已启动" }
    else { Warn "Redis 启动失败，后端将降级为内存记忆（仅开发可用）" }
} else {
    Warn "未检测到 Docker，跳过 Redis（后端自动降级为内存记忆，仅开发可用）"
}

# ---- 2. 环境变量 ----
Step "检查环境变量 backend\.env"
$envFile = Join-Path $root "backend\.env"
if (Test-Path $envFile) {
    Ok ".env 已存在"
} else {
    $example = Join-Path $root "backend\.env.example"
    if (Test-Path $example) {
        Copy-Item $example $envFile
        Warn "已从 .env.example 创建 backend\.env，请填入 LLM_API_KEY 后重新运行本脚本"
    } else {
        Warn "缺少 backend\.env，后端可能无法连接 LLM"
    }
}

# ---- 3. Python 与后端依赖 ----
Step "检查后端依赖"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Err "未找到 python，请先安装 Python 并加入 PATH"
    exit 1
}
& python -c "import uvicorn, fastapi" 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Ok "依赖已就绪"
} else {
    Warn "依赖缺失，正在安装 requirements.txt ..."
    & python -m pip install -r (Join-Path $root "backend\requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Err "依赖安装失败，请手动执行：pip install -r backend\requirements.txt"
        exit 1
    }
    Ok "依赖安装完成"
}

# ---- 4. 启动后端（独立窗口，便于查看日志/热重载）----
Step "启动后端 (Uvicorn)"
Start-Process -FilePath $psExe -WorkingDirectory (Join-Path $root "backend") `
    -ArgumentList '-NoExit', '-Command', 'uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload'
Ok "后端窗口已打开，API 文档：http://localhost:8000/docs"

# ---- 5. 启动前端（独立窗口）----
Step "启动前端 (静态服务器)"
Start-Process -FilePath $psExe -WorkingDirectory (Join-Path $root "frontend") `
    -ArgumentList '-NoExit', '-Command', 'python -m http.server 5173'
Ok "前端窗口已打开"

# ---- 6. 健康检查 ----
Step "等待服务就绪"
Wait-Http "http://localhost:8000/api/v1/health" 30 "后端"
Wait-Http "http://localhost:5173" 15 "前端"

Write-Host ""
Write-Host "启动完成：" -ForegroundColor Green
Write-Host "  前端页面：http://localhost:5173"
Write-Host "  后端 API：http://localhost:8000/docs"
Write-Host "  停止服务：关闭弹出的后端/前端窗口即可"
