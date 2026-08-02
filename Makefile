.PHONY: venv install run clean freeze help

VENV = venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip
STREAMLIT = $(VENV)/Scripts/streamlit

help:
	@echo Available commands:
	@echo   make venv      - Buat virtual environment baru
	@echo   make install   - Install semua dependencies dari requirements.txt
	@echo   make run       - Jalankan aplikasi Streamlit
	@echo   make freeze    - Update requirements.txt dari package yang terinstall
	@echo   make clean     - Hapus virtual environment

venv:
	python -m venv $(VENV)
	@echo Virtual environment berhasil dibuat. Jalankan: make install

install:
	$(PIP) install -r requirements.txt
	@echo Dependencies berhasil diinstall!

run:
	$(STREAMLIT) run streamlit_app.py

freeze:
	$(PIP) freeze > requirements.txt
	@echo requirements.txt berhasil diupdate!

clean:
	rmdir /s /q $(VENV)
	@echo Virtual environment dihapus.
