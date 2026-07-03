import os
import subprocess
import sys
import shutil

# 컴파일러 기본 검색 경로 정의
MIKTEX_PATH = r"C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64\xelatex.exe"
PANDOC_PATH = r"C:\Users\user\AppData\Local\Microsoft\WinGet\Packages\JohnMacFarlane.Pandoc_Microsoft.Winget.Source_8wekyb3d8bbwe\pandoc-3.10\pandoc.exe"

def find_executable(name, default_path):
    if os.path.exists(default_path):
        return default_path
    path = shutil.which(name)
    if path:
        return path
    return None

def run_command(cmd, cwd):
    print(f"Running command: {' '.join(cmd)} in {cwd}")
    res = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
    if res.returncode != 0:
        print(f"Warning/Error running command: {res.stderr}")
    return res.returncode == 0

def publish_all(book_type="basic"):
    xelatex = find_executable("xelatex", MIKTEX_PATH)
    pandoc = find_executable("pandoc", PANDOC_PATH)
    
    if not xelatex:
        print("Error: xelatex.exe를 찾을 수 없습니다. MiKTeX 설치 여부를 확인하십시오.")
        sys.exit(1)
    if not pandoc:
        print("Error: pandoc.exe를 찾을 수 없습니다. Pandoc 설치 여부를 확인하십시오.")
        sys.exit(1)
        
    languages = ["ko", "en", "ja"]
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "book"))
    
    print(f"====== {book_type.upper()} 도서 출판 빌드 개시 ======")
    for lang in languages:
        lang_dir = os.path.join(base_dir, lang)
        if not os.path.exists(lang_dir):
            print(f"Skipping {lang}: 폴더가 존재하지 않음 ({lang_dir})")
            continue
            
        print(f"\n--- [{lang.upper()}] 빌드 시작 ---")
        
        # 1. 고해상도 PDF 빌드 (2회 컴파일)
        jobname = f"make_llm_{book_type}-{lang}-highmarupress"
        pdf_cmd = [xelatex, f"-jobname={jobname}", "main.tex"]
        
        print("PDF 컴파일 1차...")
        run_command(pdf_cmd, lang_dir)
        print("PDF 컴파일 2차 (목차 및 참조 링크 확정)...")
        run_command(pdf_cmd, lang_dir)
        
        # 2. EPUB 전자책 빌드 via Pandoc
        epub_filename = f"make_llm_{book_type}-{lang}.epub"
        title = f"Make LLM {book_type.capitalize()} ({lang.upper()})"
        author = "dirmich"
        
        epub_cmd = [
            pandoc, "main.tex",
            "-o", epub_filename,
            "--metadata", f"title={title}",
            "--metadata", f"author={author}"
        ]
        print("EPUB 컴파일...")
        run_command(epub_cmd, lang_dir)
        
    print("\n====== 출판 빌드 완료 ======")

if __name__ == "__main__":
    # 실행 폴더 구조에 맞춰 basic 인지 advanced 인지 파악
    cwd = os.getcwd()
    book_type = "advanced" if "advanced" in cwd.lower() else "basic"
    publish_all(book_type)
