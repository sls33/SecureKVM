# SecureKVM
## Introduction
SecureKVM is a branch of SecMML that focuses on enabling secure and efficient inference (or training) for key-value models. Currently, we only support key-value models like **Markov Model** and **Decision Tree Model**.

## Running
### Network Configuration

##### Default setting

IP:port

127.0.0.1:1234

> In the default configuration, multiple processes run on a server, which is a **LAN** environment.
>
> In order to simulate **WAN** environment,  we can use the following command.

```shell
# simulate WAN
sudo tc qdisc add dev lo root netem delay 20ms rate 40Mbps
# cancle the simulation
sudo tc qdisc del dev lo root netem delay 20ms rate 40Mbps
# tc qdisc add dev <dev> root <qdisc> <qdisc-param>
```

##### How to configure

SecureKVM/util/SocketManager.cpp line127

```c++
void SocketManager::SMMLF::init() {
    init("127.0.0.1", 1234); // ip, port
}
```



### Compile

**Constant.h**

```c++
#define M 3
#define OFFLINE_PHASE_ON true
#define TOTAL_SAMPLE_SIZE 10
```

* M: the n in n-party
* OFFLINE_PHASE_ON: whether running offline phase or not
* TOTAL_SAMPLE_SIZE: the number of tests

**CMakeLists.txt**

explanation of executable file names

* `test_xxx`: test benchmark of operation ( SecGetPer Log)
* `xxx_func`: test function of basic operation (Reshare GenPerm ShuffleData SecGetPer Log)
* `DT_MPC`: Secure Decision Tree Inference
* `evalPW_mpc`: Secure Markov Inference

Compile the project at the directory SecureKVM/

```shell
cmake .
make -j
```



### Test Basic Operation

`Constant.h`

* PREDICTION_BATCH_BENCHMARK: the bacth size

Run

```shell
./executable_filename partynum
```

For example, to test protocol secgetper in three-party setting, you need open three terminals and execute the following commands **in turn**.

```shell
./test_secgetper 0
./test_secgetper 1
./test_secgetper 2
```

>partynum should start from 0, end at M-1.
>
>Please make sure the M in constant.h is right.
>
>If you want to test the correctness of the protocol, please see the file SecureKVM/test_op/test_genperm.cpp as example.



### Test Inference

`Constant.h`

set **BENCHMARK** to false;

#### 1 Secure Decision Tree Inference

`Constant.h` 

if **PERMUTE_USING_BASIC** is true, use Secget_Basic protocol, else using Secget_Fold protocol.

**FEATURE_DIM**: the feature size of dataset.

**DEPTH**: the depth of decision tree.

**PREDICTION_BATCH**: the batch size of inference

```c++
#define FEATURE_DIM 9
#define PREDICTION_BATCH 10
#define DEPTH 8
#define NODE_NUM ((1 << DEPTH) - 1)
#define PERMUTE_USING_BASIC true
#if (PERMUTE_USING_BASIC)
    #define FEATURE_VECTOR_SIZE FEATURE_DIM
#else
    #define FEATURE_VECTOR_SIZE (2 * ceil(sqrt(FEATURE_DIM)))
#endif
```

Run

```shell
./DT_MPC 0
./DT_MPC 1
...
./DT_MPC n-1
```



#### 2 Secure Markov Inference

`Constant.h`

INFER_BATCH: batch size of SecMarkov


choosing different secget protocol by param  USING_BASIC and  USING_FOLD.

| USING_BASIC         | true         | false       | false       |
| ------------------- | ------------ | ----------- | ----------- |
| USING_FOLD          | false        | true        | false       |
| the secget protocol | Secget_Basic | Secget_Fold | Secget_Perm |

* 2-gram 

```c++
#define Markov_INFER_LOG_BATCH 1000
#define CP_DOMAIN_SIZE ((ALPHABET_SIZE*ALPHABET_SIZE) + 1)
#define IP_DOMAIN_SIZE (ALPHABET_SIZE)
```

* 3-gram

```c++
#define Markov_INFER_LOG_BATCH 10000
#define CP_DOMAIN_SIZE ((ALPHABET_SIZE*ALPHABET_SIZE * ALPHABET_SIZE) + 1)
#define IP_DOMAIN_SIZE (ALPHABET_SIZE*ALPHABET_SIZE)
```

`run_Markov_eval.sh`

* 2-gram: model_n="2"
* 3-gram: model_n="3"

```shell
./run_Markov_eval.sh 0
./run_Markov_eval.sh 1
...
./run_Markov_eval.sh n-1
```

## Help

Any question, please contact 19212010008@fudan.edu.cn.

## Contributor

**Faculty**: Prof. Weili Han

**Students**: Haoqi Wu (Graduate Student), Zifeng Jiang (Graduate Student), Wenqiang Ruan (Ph.D Candidate), Shuyu Chen (Graduate Student), Xinyu Tu (Graduate Student), Zhexuan Wang (Graduate Student), Lushan Song (Ph.D Candidate), Dingyi Tang (Post Graduate Student)
