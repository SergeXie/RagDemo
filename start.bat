@echo off
chcp 936 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo    RagDemo 一键启动
echo ============================================

rem ---- 1. Redis (可选) ----
echo.
echo [1/6] 启动 Redis
where docker >nul 2>nul
if errorlevel 1 goto no_docker
docker compose up -d redis >nul 2>nul
if not errorlevel 1 goto redis_ok
docker-compose up -d redis >nul 2>nul
if not errorlevel 1 goto redis_ok
echo   [提示] Redis 启动失败，后端将降级为内存记忆（仅开发可用）
goto redis_done
:redis_ok
echo   [OK] Redis 容器已启动
goto redis_done
:no_docker
echo   [提示] 未检测到 Docker，跳过 Redis（后端自动降级为内存记忆）
:redis_done

rem ---- 2. 环境变量 ----
echo.
echo [2/6] 检查环境变量 backend\.env
if not exist "backend\.env" goto env_missing
echo   [OK] .env 已存在
goto env_done
:env_missing
if not exist "backend\.env.example" goto env_noexample
copy /y "backend\.env.example" "backend\.env" >nul
echo   [提示] 已从 .env.example 创建 backend\.env，请填入 LLM_API_KEY 后重新运行本脚本
goto env_done
:env_noexample
echo   [提示] 缺少 backend\.env，后端可能无法连接 LLM
:env_done

rem ---- 3. 依赖 ----
echo.
echo [3/6] 检查后端依赖
where python >nul 2>nul
if errorlevel 1 goto no_python
python -c "import uvicorn, fastapi" >nul 2>nul
if errorlevel 1 goto install_deps
echo   [OK] 依赖已就绪
goto deps_done
:install_deps
echo   [提示] 依赖缺失，正在安装 requirements.txt ...
python -m pip install -r "backend\requirements.txt"
if errorlevel 1 goto pip_fail
echo   [OK] 依赖安装完成
goto deps_done
:pip_fail
echo   [错误] 依赖安装失败，请手动执行: pip install -r backend\requirements.txt
goto fail
:no_python
echo   [错误] 未找到 python，请先安装 Python 并加入 PATH
goto fail
:deps_done

rem ---- 4. 后端 ----
echo.
echo [4/6] 启动后端 (Uvicorn, 端口 8000)
start "RagDemo-Backend" /d "%~dp0backend" cmd /k "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo   [OK] 后端窗口已打开: http://localhost:8000/docs

rem ---- 5. 前端 ----
echo.
echo [5/6] 启动前端 (端口 5173)
start "RagDemo-Frontend" /d "%~dp0frontend" cmd /k "python -m http.server 5173"
echo   [OK] 前端窗口已打开

rem ---- 6. 健康检查 ----
echo.
echo [6/6] 等待服务就绪
where curl >nul 2>nul
if errorlevel 1 goto no_curl
set TRY=0
:wait_backend
ping -n 2 127.0.0.1 >nul
set /a TRY+=1
curl -s -o nul --connect-timeout 2 "http://localhost:8000/api/v1/health" >nul 2>nul
if not errorlevel 1 goto backend_ok
if %TRY% geq 30 goto backend_late
goto wait_backend
:backend_late
echo   [提示] 后端暂未响应，请查看后端窗口输出
goto backend_ok
:backend_ok
echo   [OK] 后端就绪: http://localhost:8000/api/v1/health
set TRY=0
:wait_frontend
ping -n 2 127.0.0.1 >nul
set /a TRY+=1
curl -s -o nul --connect-timeout 2 "http://localhost:5173/" >nul 2>nul
if not errorlevel 1 goto frontend_ok
if %TRY% geq 15 goto frontend_late
goto wait_frontend
:frontend_late
echo   [提示] 前端暂未响应，请查看前端窗口输出
goto frontend_ok
:frontend_ok
echo   [OK] 前端就绪: http://localhost:5173
goto summary
:no_curl
echo   [提示] 未找到 curl，跳过健康检查
:summary
echo.
echo 启动完成:
echo   前端页面: http://localhost:5173
echo   后端 API: http://localhost:8000/docs
echo   停止服务: 关闭后端/前端窗口即可
echo.
pause
endlocal
exit /b 0

:fail
echo.
echo 启动失败，请根据上方错误提示处理。
echo.
pause
endlocal
exit /b 1
