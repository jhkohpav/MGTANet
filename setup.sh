cd det3d/ops/dcn 
python setup.py build_ext --inplace

cd .. && cd iou3d_nms
python setup.py build_ext --inplace

cd .. && cd defromDETR
python setup.py build install

cd ../../../apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
