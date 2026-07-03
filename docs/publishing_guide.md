# 아마존 KDP 출판 및 일괄 빌드 가이드 문서

이 문서는 **Make LLM Advanced** 도서 시리즈의 다국어판(ko, en, ja)을 아마존 KDP(Kindle Direct Publishing) 판매 규격에 맞게 일괄 빌드(PDF 및 EPUB)하고 관리하는 가이드입니다.

---

## 1. 선행 요구사항 (Prerequisites)

도서를 정상적으로 컴파일하기 위해서는 다음 도구들이 시스템에 설치되어 있어야 합니다.

### 1-1. XeLaTeX (MiKTeX) 설치
- 다국어(한국어, 영어, 일본어) 및 수학 수식을 고해상도로 컴파일하기 위해 `XeLaTeX`가 필요합니다.
- Windows 환경에서는 **MiKTeX**을 설치하는 것을 권장합니다.
  - [MiKTeX 공식 홈페이지](https://miktex.org/)에서 다운로드하여 기본 경로에 설치합니다.

### 1-2. Pandoc 설치
- LaTeX 소스코드(`.tex`)를 표준 규격의 전자책 포맷인 `EPUB` 파일로 변환하기 위해 `Pandoc`이 필요합니다.
- Windows 10/11 시스템 환경의 경우, 터미널에서 다음 `winget` 명령어로 무소음(Silent) 자동 설치할 수 있습니다:
  ```powershell
  winget install -e --id JohnMacFarlane.Pandoc --force --accept-package-agreements --accept-source-agreements
  ```

---

## 2. 출판 빌드 명령어 (Build Guide)

프로젝트 루트 디렉토리에서 자동화 빌드 스크립트를 실행하면 3개 국어(한국어, 영어, 일본어)의 PDF 및 EPUB 산출물을 일괄 자동 추출합니다:

```powershell
# 프로젝트 루트 디렉토리에서 실행
python tools/publish_book.py
```

### 빌드 메커니즘
1. **XeLaTeX 컴파일 (2회)**: 페이지 번호, 참조 링크 및 목차 정보를 정확하게 결정하기 위해 `xelatex`를 2회 연속 수행하여 `-highmarupress.pdf` 고해상도 PDF 파일을 생성합니다.
2. **Pandoc EPUB 컴파일**: `main.tex`를 기준으로 여러 장(Chapter) 소스들을 완전한 호환성을 갖춘 하나의 `.epub` 전자책 파일로 포장합니다.

---

## 3. 최종 산출물 상세 명세

빌드 완료 시 `book/` 폴더 내의 각 언어별 디렉토리에 다음 파일들이 생성됩니다:

* **한국어판 (`book/ko/`)**
  - 인쇄용 PDF: `make_llm_advanced-ko-highmarupress.pdf`
  - 전자책용 EPUB: `make_llm_advanced-ko.epub`
* **영어판 (`book/en/`)**
  - 인쇄용 PDF: `make_llm_advanced-en-highmarupress.pdf`
  - 전자책용 EPUB: `make_llm_advanced-en.epub`
* **일본어판 (`book/ja/`)**
  - 인쇄용 PDF: `make_llm_advanced-ja-highmarupress.pdf`
  - 전자책용 EPUB: `make_llm_advanced-ja.epub`

---

## 4. 중요 보안 정책 (Git 유출 차단)

> [!WARNING]
> - 도서의 LaTeX 원고 원본(`book/` 폴더) 및 컴파일된 최종 PDF, EPUB 파일들은 **저작권 보호를 위해 절대 GitHub 원격 저장소에 노출되어서는 안 됩니다.**
> - 본 프로젝트의 `.gitignore` 파일에 다음 차단 정책이 기본 수립되어 있으므로 안심하고 로컬에서 빌드 작업을 진행하셔도 좋습니다:
>   ```
>   # 책 원고 및 출판 파일 차단 (보안)
>   book/
>   *.pdf
>   *.epub
>   ```
