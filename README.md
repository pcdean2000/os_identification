# OS Identification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple project for identifying the operating system of a remote host based on network traffic analysis (e.g., TCP/IP fingerprinting).

## Description

This project provides a script to passively parse network traffic to determine the OS of target machines. It leverages specific characteristics of network packets (like TTL values, window sizes, and TCP options) furhter with proposed statistical features that differ between operating systems.

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/pcdean2000/os_identification.git](https://github.com/pcdean2000/os_identification.git)
    cd os_identification
    ```

2.  建立並啟用一個虛擬環境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  安裝所需的依賴套件:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python main.py
```

## License

This project is licensed under the MIT License - see the `LICENSE.md` file for details.