#!/bin/bash
# download_datasets.sh
#
# This script automatically downloads and extracts the raw datasets
# for the project. It checks if the target directory already contains
# files before downloading.
#
# Dependencies: curl, unzip

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---

# Base directory for raw data, relative to the script's location
# Assumes this script is run from the project root.
DATA_RAW_DIR="Data/raw"

USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
ACCEPT_HEADER="text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
ACCEPT_LANG_HEADER="en-US,en;q=0.9,zh;q=0.8"


# Associative array for dataset names and their URLs
# Using declare -A for bash 4.0+
declare -A DATASET_URLS
DATASET_URLS["lasto18"]="https://is.muni.cz/repo/1402160/anonymized_flow.zip"
DATASET_URLS["lasto20"]="https://zenodo.org/records/3461771/files/flows_dataset_anonymized.zip"
DATASET_URLS["lasto23"]="https://zenodo.org/records/7635138/files/anonymized_flows.zip"

# --- Helper Function ---

# 自動扁平化函式
# 檢查一個目錄是否只包含一個子目錄，如果是，則將內容上移
flatten_if_single_subdir() {
    local target_dir="$1"
    echo "Checking for nested directory structure in $target_dir..."

    local items_in_target=()
    # 使用 find 安全地獲取頂層項目（檔案和目錄）
    # -print0 和 read -d '' 組合可以處理帶有空格或特殊字元的檔名
    while IFS= read -r -d '' item; do
        items_in_target+=("$item")
    done < <(find "$target_dir" -mindepth 1 -maxdepth 1 -print0)
    
    local item_count=${#items_in_target[@]}

    if [ "$item_count" -eq 1 ]; then
        local single_item="${items_in_target[0]}"
        if [ -d "$single_item" ]; then
            # 找到了一個單一目錄
            echo "Found a single directory: $single_item. Flattening..."
            
            # 在子 shell 中啟用 dotglob，這樣 '*' 就能匹配隱藏檔
            # 然後將子目錄的所有內容移動到父目錄
            (
                shopt -s dotglob
                mv "$single_item"/* "$target_dir"
            )
            
            # 移除現在已經空了的子目錄
            rmdir "$single_item"
            echo "Flattened '$single_item' into '$target_dir'."
        else
            echo "Found a single file. No flattening needed."
        fi
    else
        echo "Found $item_count items. No flattening needed."
    fi
}


# --- Main Logic ---

echo "===== Starting Dataset Download ====="

# Ensure the base raw data directory exists
mkdir -p "$DATA_RAW_DIR"

# Loop through each dataset defined in the array
for name in "${!DATASET_URLS[@]}"; do
    url="${DATASET_URLS[$name]}"
    target_dir="$DATA_RAW_DIR/$name"

    # 1. Create target directory
    mkdir -p "$target_dir"

    # 2. Check if directory is empty
    #    ls -A returns true (exit code 0) if files exist
    if [ "$(ls -A "$target_dir")" ]; then
        echo "Dataset '$name' already exists in $target_dir. Skipping."
    else
        echo "Downloading '$name' from: $url..."
        
        # 3. Download and extract
        temp_zip_file="$target_dir/temp_download.zip"
        
        # 這是所有請求共用的基礎標頭
        base_curl_args=("-L"
                       "--compressed" # 自動處理 Accept-Encoding
                       "-A" "$USER_AGENT"
                       "-H" "$ACCEPT_HEADER"
                       "-H" "$ACCEPT_LANG_HEADER"
                       "-H" "sec-ch-ua: \"Microsoft Edge\";v=\"141\", \"Not?A_Brand\";v=\"8\", \"Chromium\";v=\"141\""
                       "-H" "sec-ch-ua-mobile: ?0"
                       "-H" "sec-ch-ua-platform: \"Windows\""
                       )
        
        # 初始化 final_curl_args 陣列
        final_curl_args=("${base_curl_args[@]}")

        # 模仿 GitHub 點擊的單一步驟
        if [ "$name" == "lasto18" ]; then
            echo "Adding GitHub Referer headers and long timeout for lasto18..."
            github_referer="https://github.com/CSIRT-MU/PassiveOSFingerprint/tree/master/Dataset"
            
            # 添加您在 GitHub 日誌中觀察到的特定標頭
            final_curl_args+=("-e" "$github_referer" # -e 是 --referer 的縮寫
                              "-H" "sec-fetch-dest: document"
                              "-H" "sec-fetch-mode: navigate"
                              "-H" "sec-fetch-site: cross-site" 
                              "-H" "sec-fetch-user: ?1"
                              "-H" "upgrade-insecure-requests: 1"
                              "--connect-timeout" "30" # 30秒連線超時
                              "-m" "180" # 180秒 (3分鐘) 總下載超時，以應對 1 分鐘的等待
                             )
        fi
        
        # 添加 URL 和輸出檔案
        final_curl_args+=("$url" "-o" "$temp_zip_file")
        
        # 使用 "${final_curl_args[@]}" 安全地執行命令
        echo "Executing curl..."
        curl "${final_curl_args[@]}"

        echo "Download complete. Extracting '$name' to $target_dir..."
        
        # 檢查下載的檔案是否為 HTML (下載失敗的標誌)
        if file "$temp_zip_file" | grep -q "HTML document"; then
            echo "ERROR: Download failed for '$name'. Server returned an HTML page (likely an error)."
            echo "Please check the URL or network and try again."
            # 顯示 HTML 內容以便除錯
            echo "--- Server Response (first 10 lines) ---"
            head -n 10 "$temp_zip_file"
            echo "----------------------------------------"
            rm "$temp_zip_file" # 刪除失敗的 HTML 檔案
            # exit 1 # 保持註解，以便繼續下載其他資料集
        else
            # 檔案看起來正常 (不是 HTML)，繼續解壓縮
            unzip "$temp_zip_file" -d "$target_dir"
            
            # Remove the temporary zip file
            rm "$temp_zip_file"

            # 呼叫通用的扁平化函式
            flatten_if_single_subdir "$target_dir"
            
            echo "'$name' has been successfully downloaded and extracted."
        fi
    fi
done

echo "===== Dataset Download Finished ====="

