# sleep_detection
sleep detection using opencv in google colab

- chạy trên laptop: 
  - cài các thư viện cần thiết (window hơi rắc rối trong việc cài dlib)
  - clone project sau đó chạy file pi_drow.py, cấu hình các đường dẫn nếu cần
- chạy trên colab: 
  - mở file sleep detection.ipynb trên google colab
  - chạy 3 kernel đầu
  
** google colab sẽ dùng video đã quay trước, không có realtime, không gặp vấn đề trong cài 
đặt thư viện. **



## Cách cài đặt và sử dụng trên win 10:
  1. Cài đặt python 3.6
  2. upgrade pip: pip install --upgrade pip
  2. Cài đặt cmake [link](https://cmake.org/download/) và add PATH cho cmake
  3. Cài đặt dlib: 
  open cmd, run command: 
  `python -m pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54bfcf`
 
  4. Cài đặt thêm các thư viện cần thiết: pip install numpy imutils playsound
  5. enjoy
