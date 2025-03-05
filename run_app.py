import os
import sys
import subprocess

# Streamlit 파일 감시 기능 비활성화
os.environ['STREAMLIT_SERVER_WATCH_DIRS'] = 'false'

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# service 디렉토리를 Python 경로에 추가
service_dir = os.path.join(current_dir, 'service')
sys.path.append(service_dir)

print(f"Added to Python path: {current_dir}")
print(f"Added to Python path: {service_dir}")

# Streamlit 앱 실행
streamlit_cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.fileWatcherType", "none"]
subprocess.run(streamlit_cmd) 