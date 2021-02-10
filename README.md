# code of STMoE
This repository contains experiments of Dynamic MNIST (Section 4.1 in main manuscript).  
Two STMoE models are shown. 

## 1. requirement  
tensorflow-gpu 1.13.1

## 2. data preparetion
```
cd data
python Dynamic_MNIST.py
```
`train_data.npy`, `valid_data.npy`, and `test_data.npy` will be generated in the `./data`.

## 3. training
Trained model parameters of experts in local optimization is `./models`.  
Training only STGN in global optimization will be done as follows:  
```
python STMoE-1_train.py --training gating
```
or  
```
python STMoE-2_train.py --training gating
```
Training the whole model in global optimization will be done as follows:
```
python STMoE-1_train.py --training all
```
or  
```
python STMoE-2_train.py --training all
```

## 4. test
Test notebook are [STMoE-1](https://github.com/papercodeanonymous755/STMoE/blob/main/STMoE-1_test.ipynb) and [STMoE-2](https://github.com/papercodeanonymous755/STMoE/blob/main/STMoE-2_test.ipynb).
Trained models are in `./models`.  
You can do only test and get the following results. 
- STMoE-1

![image](https://user-images.githubusercontent.com/78733182/107466007-ba57ee00-6ba6-11eb-834a-01e485c1815c.png)
![image](https://user-images.githubusercontent.com/78733182/107466030-c9d73700-6ba6-11eb-8a10-2561808c836d.png)

- STMoE-2

![image](https://user-images.githubusercontent.com/78733182/107465838-749b2580-6ba6-11eb-9765-510f698f3d0d.png)
![image](https://user-images.githubusercontent.com/78733182/107465920-95637b00-6ba6-11eb-896a-daec23f4d391.png)

## 5. References
- MIM https://github.com/Yunbo426/MIM
- Robust Function https://github.com/google-research/google-research/tree/master/robust_loss
