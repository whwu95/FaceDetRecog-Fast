1.cmake caffe:
  cd caffe
  mkdir cmake_build
  cd cmake_build
  cmake ..
  make && make install
2.cmake face code 
  cd face_code
  mkdir build 
  cd build 
  cmake ..
  make
3.run
  cd face_code  
-----ExtractFeature
./build/ExtractFeature lfw_list.txt 13233 lfwFeature.txt lfwtName.txt
./build/ExtractFeature casia_list.txt 20000 casiaFeature.txt casiaName.txt

------1 Feature recog
./build/FaceRecog1 lfwFeature.txt lfwtName.txt resultLfwRecog1.txt
./build/FaceRecog1 casiaFeature.txt casiaName.txt resultCasiaRecog1.txt


--------veri
./build/ExtractFeature casia_db_list.txt 6000 casiaDbFeature.txt casiaDbName.txt
./build/ExtractFeature casia_test_list.txt 6000 casiaTestFeature.txt casiaTestName.txt
./build/FaceVeri casiaDbFeature.txt casiaTestFeature.txt 3000 resultCasiaVeri.txt

./build/ExtractFeature lfw_db_list.txt 6000 lfwDbFeature.txt lfwDbName.txt
./build/ExtractFeature lfw_test_list.txt 6000 lfwTestFeature.txt lfwTestName.txt
./build/FaceVeri lfwDbFeature.txt lfwTestFeature.txt 3000 resultLfwVeri.txt
