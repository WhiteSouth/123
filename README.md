
[toc]


## 概述

**运行环境**

- Ubuntu 18.04 LTS


**支持 (详见下文)**

- Balance Basic PSI
- UnBalance Basic PSI
- UnBalance Label PSI
- MPC-3PC-训练
- MPC-2PC-训练
- MPC-2PC-预测
- GBDT-2PC-预测

**测试数据**

- PSI/MPC 采用自生成数据 (如何生成与使用，详见下文)。
- GBDT 的数据来自 [鸢尾花](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)。


## PSI

**角色说明**

共有两方，

- ALICE，称被查询方，即 sender。
- BOB，称查询方，即 receiver。


### PSI Fake 数据生成

进入任何一个 `PSI` 相关 `bin/` 目录，

```
bin# ./gen_fake_data_mt_simple 
Usages:
	./gen_fake_data_mt_simple <size[100]> <percent for receiver[3]> <labels[9]>
```

3 个参数依次是 `sender 方数据量`，`receiver 方占 sender 方数据量的比值(%)`，`label 的数量`。


### DH-Balance-Basic-PSI

> project: psi-balance-basic

- 适⽤于两方数据量对等均衡的情况
- 适⽤于训练中的样本/标签对⻬


|        |                                                 |
| ------ | ----------------------------------------------- |
| 环境   | 4C-16G-100M/24C-64G-LAN                         |
| 数据量 | (100K-100K) / (1M-1M) / (10M-10M) / (100M-100M) |
| 指标   | 时间/通信量                                     |


**使用步骤：**

进入 `bin/`。


```bash
cd bin
export LD_LIBRARY_PATH=../lib
```

**一，生成数据。**

- 创建 `data` 目录并进入，使用 `gen_fake_data_mt_simple` 生成测试需要的 fake 数据 (被查询方 1000条，查询方是被查询方的 80%，两方数据均衡），然后返回 `bin` 目录。

```bash
mkdir -p data
cd data
../gen_fake_data_mt_simple 1000 80
cd ..
```

执行上面生成 fake 数据命令，将生成 4 个文件，它们分别为：1000-receiver.csv，1000-receiver-bal.csv，1000-sender.csv，1000-sender_only_id.csv。

1. 对于 Balance Basic PSI 来说：
Sender 端使用的输入数据是：1000-sender_only_id.csv
Receiver 端使用的输入数据是：1000-receiver-bal.csv

2. 对于 Unbalance Basic PSI 来说：
Sender 端使用的输入数据是：1000-sender_only_id.csv
Receiver 端使用的输入数据是：1000-receiver.csv

3. 对于 Unbalance Label PSI 来说：
Sender 端使用的输入数据都是：1000-sender.csv
Receiver 端使用的输入数据都是：1000-receiver.csv

**二，执行。**

- 执行前，先看一下程序的帮助信息。

```
./psi-balance-basic --help
usage: ./psi-balance-basic --partyid=int [options] ...
options:
  -h, --help             help (string [=this help list])
  -p, --partyid          0:ALICE;1:BOB (int)
      --input_id         input ids path (string [=input_id.csv])
      --result_output    result output path (string [=result.csv])
      --self_port        self port (int [=14196])
      --peer_host        peer host/ip (string [=127.0.0.1])
      --peer_port        peer port (int [=15225])
      --recv_party       0:ALICE;1:BOB;2:BOTH(ALICE and BOB) (int [=0])
      --thread_num       thread number, default 4 (int [=4])

For example:
   ./psi-balance-basic --partyid=1 --self_port=11888 --peer_host=127.0.0.1 --peer_port=22666 \
       --recv_party=2 --input_id=phone_numberb.txt --result_output=phone_number_outb.txt
   ./psi-balance-basic --partyid=0 --self_port=22666 --peer_host=127.0.0.1 --peer_port=11888 \
       --recv_party=2 --input_id=phone_numbera.txt --result_output=phone_number_outa.txt
```


- 开始求交集。（ALICE/BOB 两个节点/终端)

ALICE (被查询方) 节点。

```bash
mkdir -p result
./psi-balance-basic --partyid=0 --thread_num=8 \
   --self_port=22666 --peer_host=127.0.0.1 --peer_port=11888 \
   --recv_party=2 --input_id=data/1000-sender_only_id.csv \
   --result_output=result/1000-res-sender.csv
```

BOB (查询方) 节点。

```bash
mkdir -p result
./psi-balance-basic --partyid=1 --thread_num=8 \
   --self_port=11888 --peer_host=127.0.0.1 --peer_port=22666 \
   --recv_party=2 --input_id=data/1000-receiver-bal.csv \
   --result_output=result/1000-res-receiver.csv
```

其中参数说明


| 参数            | 描述                                       |
| --------------- | ------------------------------------------ |
| `partyid`       | 指定当前是哪一方。                         |
| `thread_num`    | 两方必须一样                               |
| `self_port`     | 当前节点的监听 PORT。                      |
| `peer_host`     | 对端节点的 IP。                            |
| `peer_port`     | 对端节点的 PORT (即对端节点的 self_port)。 |
| `recv_party`    | 哪一方（或两方）取得结果。                 |
| `result_output` | 结果（交集）保存路径                       |



**三，结果输出。**

最后结果会保存在 `--result_output` 指定的文件中（如果 `--recv_party` 包含了该节点的话）。



### DH-Unbalanced-Basic-PSI

> project: psi-unbalance-basic


- 适⽤于标签数量不对称查询 (查询方标签数量少，被查询方标签数量多)


|        |                                                            |
| ------ | ---------------------------------------------------------- |
| 环境   | 4C-16G-100M/24C-64G-LAN                                    |
| 数据量 | (100K-10) / (1M-100) / (10M-1K) / (100M-10K) / (1B - 100K) |
| 指标   | 时间/通信量                                                |


```
root@vm:~/x/check/psi-unbalance-basic/bin# export LD_LIBRARY_PATH=../lib
root@vm:~/x/check/psi-unbalance-basic/bin# ./psi-unbalance-basic --help
usage: ./psi-unbalance-basic --partyid=int [options] ...
options:
  -h, --help              help (string [=this help list])
  -p, --partyid           0:ALICE(SENDER);1:BOB(RECEIVER) (int)
      --psi_type          PSI type(Basic,HeBasic,Label) (string [=Label])
      --sender_input      sender input ids (and labels) path (string [=sender_ids_labels.csv])
      --receiver_input    receiver input ids path (string [=receiver_ids.csv])
      --result_output     result output path (string [=result.csv])
      --host              server(alice) host/ip (string [=127.0.0.1])
      --port              server(alice) port (int [=15225])
      --run_times         how many run phases will be executed (int [=1])
      --log_prefix        the log prefix path for cpp (string [=psi_sm])

For example:
   ./psi-unbalance-basic --partyid=0 --psi_type=Basic --sender_input=sender_ids_labels.csv
   ./psi-unbalance-basic --partyid=1 --psi_type=Basic --receiver_input=receiver_ids.csv --result_output=result.csv
```

使用方式同 [DH-Balance-Basic-PSI](#dh-balance-basic-psi)。

**注意**，在使用上，指定 `--psi_type=Basic`。（`HeBasic` 暂不支持）

与 `DH-Balance-Basic-PSI` 的区别是，这里的 `host/port` 需要填写 `ALICE` 节点信息。

输入通过 `--sender_input/--receiver_input` 指定。

输出通过 `--result_output` 指定。（只对 receiver 方有效）


### DH-Unbalanced-Labeled-PSI

> project: psi-unbalance-label

- 适⽤于标签数量不对称查询 (查询方标签数量少，被查询方

|        |                                                            |
| ------ | ---------------------------------------------------------- |
| 环境   | 4C-16G-100M/24C-64G-LAN                                    |
| 数据量 | (100K-10) / (1M-100) / (10M-1K) / (100M-10K) / (1B - 100K) |
| 标签数 | 10                                                         |
| 指标   | 时间/通信量                                                |


```
root@vm:~/x/check/psi-unbalance-label/bin# export LD_LIBRARY_PATH=../lib
root@vm:~/x/check/psi-unbalance-label/bin# ./psi-unbalance-label --help
usage: ./psi-unbalance-label --partyid=int [options] ...
options:
  -h, --help              help (string [=this help list])
  -p, --partyid           0:ALICE(SENDER);1:BOB(RECEIVER) (int)
      --psi_type          PSI type(Basic,HeBasic,Label) (string [=Label])
      --sender_input      sender input ids (and labels) path (string [=sender_ids_labels.csv])
      --receiver_input    receiver input ids path (string [=receiver_ids.csv])
      --result_output     result output path (string [=result.csv])
      --host              server(alice) host/ip (string [=127.0.0.1])
      --port              server(alice) port (int [=15225])
      --run_times         how many run phases will be executed (int [=1])
      --log_prefix        the log prefix path for cpp (string [=psi_sm])

For example:
   ./psi-unbalance-label --partyid=0 --psi_type=Label --sender_input=sender_ids_labels.csv
   ./psi-unbalance-label --partyid=1 --psi_type=Label --receiver_input=receiver_ids.csv --result_output=result.csv
```

使用方式同 [DH-Unbalanced-Basic-PSI](#dh-unbalanced-basic-psi)。

注意，在使用上，这里与 `DH-Unbalanced-Basic-PSI` 的唯一区别是 `--psi_type=Label`。


## Machine Learning


### 2PC/3PC Fake 数据生成

进入任何一个 `mpc` 相关 `bin/` 目录，

```
bin# python ./bench_data_gen.py --help
usage: bench_data_gen.py [-h] [--f F] [--s S] [--p P]

    bench_data_gen.py --f=200 --s=100000 --p=3
    
    Generating standard plaintext datasets for MPC benchmark
    用于临时生成 s*f 大小的标准分类数据集、回归数据集，且进行p方之间的水平、垂直划分，以进行MPC的评测. （请不要用于其他目的）

    参数：
        f：特征维度数
        s：基础样本数，如设置为10万，将生成10万、20万、30万、50万、100万数据集;
        p: 支持设置为2或3, 设置为3时可以用于3PC的3方私有数据输入场景；设置为2时可以用于2PC的两方私有数据输入场景或2PC+helper式的场景。

    结果：
        将在benchmark目录下生成对应的各种数据集。
        * benchmark/cls
            此目录存放生成的2分类标准数据集
            benchmark/cls/XXXX 
                此目录存放总数据大小为XXXX的数据集, 训练数据集和测试数据集按4:1比例划分
                train_X.csv
                    完整训练数据集中的feature数据
                train_Y.csv
                    完整训练数据集中的分类label数据
                pred_X.csv
                    完整验证数据集中的feature数据
                pred_Y.csv
                    完整验证数据集中的目录label数据
                
                train_X_V_partyA.csv, train_X_V_partyB.csv, train_X_V_partyC.csv
                    按feature均匀切分的2或3方训练数据集中的feature数据
                    (p设置为2时，train_X_V_partyC.csv内容为空)
                pred_X_V_partyA.csv, pred_X_V_partyB.csv, pred_X_V_partyC.csv
                    按feature均匀切分的2或3方验证数据集中的feature数据
                    (p设置为2时，train_X_V_partyC.csv内容为空)

                train_X_H_partyA.csv, train_X_H_partyB.csv, train_X_H_partyC.csv
                    按sample均匀切分的2或3方训练数据集中的feature数据
                    (p设置为2时，train_X_H_partyC.csv内容为空)
                train_Y_H_partyA.csv, train_Y_H_partyB.csv, train_Y_H_partyC.csv
                    按sample均匀切分的2或3方训练数据集中的label数据
                    (p设置为2时，train_Y_H_partyC.csv内容为空)                
                pred_X_H_partyA.csv, pred_X_H_partyB.csv, pred_X_H_partyC.csv
                    按sample均匀切分的2或3方验证数据集中的feature数据
                    (p设置为2时，train_X_H_partyC.csv内容为空)             
                pred_Y_H_partyA.csv, pred_Y_H_partyB.csv, pred_Y_H_partyC.csv
                    按sample均匀切分的2或3方验证数据集中的feature数据
                    (p设置为2时，train_Y_H_partyC.csv内容为空)                  

                empty_input.csv 
                    空的置位文件,请不要删除我
                
                train_Y_target.csv, pred_Y_target.csv
                    辅助文件，用于精度评估
        * benchmark/reg
            此目录存放生成的线性回归标准数据集，与cls目录同构，不再赘述.    

optional arguments:
  -h, --help  show this help message and exit
  --f F       feature number
  --s S       basic sample number
  --p P       default parts to split
```

比如，现在要生成基础样本数为100，特征数为10的三方数据用于测试。

```bash
python ./bench_data_gen.py --s 100 --f 10 --p 3
```

终端输出

```
Namespace(f=10, p=3, s=100)

 n_samples = 100, n_features = 10:

 n_samples = 200, n_features = 10:

 n_samples = 300, n_features = 10:

 n_samples = 500, n_features = 10:

 n_samples = 1000, n_features = 10:
###############Generating Regression data DONE!###############
[reg_f10_s100]LinReg           0.99813017
[reg_f10_s100]SGDR             0.99814925

[reg_f10_s200]LinReg           0.99346269
[reg_f10_s200]SGDR             0.99345362

[reg_f10_s300]LinReg           0.99654162
[reg_f10_s300]SGDR             0.99653894

[reg_f10_s500]LinReg           0.99420946
[reg_f10_s500]SGDR             0.99421288

[reg_f10_s1000]LinReg          0.99564888
[reg_f10_s1000]SGDR            0.99564774

###############Analysis Regression data DONE!###############

 n_samples = 100, n_features = 10:

 n_samples = 200, n_features = 10:

 n_samples = 300, n_features = 10:

 n_samples = 500, n_features = 10:

 n_samples = 1000, n_features = 10:
###############Generating Classification data DONE!###############
[clf_f10_s100]LR               0.85
[clf_f10_s100]SGDC             0.85
[clf_f10_s200]LR               0.95
[clf_f10_s200]SGDC             0.825
[clf_f10_s300]LR               0.95
[clf_f10_s300]SGDC             0.95
[clf_f10_s500]LR               0.94
[clf_f10_s500]SGDC             0.93
[clf_f10_s1000]LR              0.955
[clf_f10_s1000]SGDC            0.955
###############analysis Classification data DONE!###############
```

我们先简单地看一下用于（测试）分类的数据 (在 `当前目录/benchmark/cls/1000`)

```
benchmark/cls/1000
├── empty_input.csv
├── pred_X.csv
├── pred_X_H_partyA.csv
├── pred_X_H_partyB.csv
├── pred_X_H_partyC.csv
├── pred_X_V_partyA.csv
├── pred_X_V_partyB.csv
├── pred_X_V_partyC.csv
├── pred_Y.csv
├── pred_Y_H_partyA.csv
├── pred_Y_H_partyB.csv
├── pred_Y_H_partyC.csv
├── pred_Y_target.csv
├── train_X.csv
├── train_X_H_partyA.csv
├── train_X_H_partyB.csv
├── train_X_H_partyC.csv
├── train_X_V_partyA.csv
├── train_X_V_partyB.csv
├── train_X_V_partyC.csv
├── train_Y.csv
├── train_Y_H_partyA.csv
├── train_Y_H_partyB.csv
├── train_Y_H_partyC.csv
└── train_Y_target.csv
```

关于上述文件命名的说明 （也可参考上面 bench_data_gen.py 的帮助信息）

```
train_* 用于训练；pred_* 用于预测；
*_X_* 表示是 feature 数据 (样本)，用于下文配置文件中 DATA_X 字段；
*_Y_* 表示是 label 数据 (TARGET)，用于下文配置文件中 DATA_Y 字段；
*_H_* 表示是横向拆分的数据；*_V_* 表示是纵向拆分的数据；
*_party[A|B|C] 表示这个文件中的数据是属于哪一方的私有数据；
*_Y_target 用于精度评估；
empty_input.csv 用于占位。
```

### 2PC/3PC 配置文件说明

字段说明

```json
{
    "MAX_EPOCHS":1, // 训练时，迭代次数，多方必须一样
    "MINI_BATCH":64,  // 训练时，每次迭代的样本数量
    "USE_KFOLD":0, // 暂时无效
    "MOMENTUM":0,  // 暂时无效
    "NESTEROV":false,  // 暂时无效
    "SAVE_MODEL":"SECURE-MYTYPE",  // 暂时无效
    "KEY_DIR":"./key/", // 暂时无效
    "DATA_DIR":"./data/", // 数据存放目录
    "DATA_X":"train_X_V_partyA.csv", // 训练或预测时的样本数据(X)
    "DATA_Y":"train_Y.csv", // 训练时的标签数据(y)
    "BASE_PORT":32000, // 暂时无效
    "LABEL_PARTY":0,  // 纵向划分数据时，标签在哪一方
    "TYPE":"3PC", // 2PC 或者 3PC
    "C_OWNS_DATA":false, // 第三方(P2)是否有数据
    "USER_FEATURE":"SameUsersDiffFeatures", // 纵向(SameUsersDiffFeatures)或横向(SameFeaturesDiffUsers)划分数据
    "P0":{ // 节点 0
      "NAME":"party name A(P0)", // 节点名称，用于显示
      "HOST":"127.0.0.1", // IP
      "PORT":[32003,31006], // 监听端口
      "KEYS":["keyA","keyAB","keyAC","null"] // 暂时无效
    },   
    "P1":{// 节点 1
      "NAME":"party name B(P1)",
      "HOST":"127.0.0.1",
      "PORT":[32001,32007],
      "KEYS":["keyB","keyAB","null","keyBC"]
    },   
    "P2":{// 节点 2
      "NAME":"party name C(P2)",
      "HOST":"127.0.0.1",
      "PORT":[42002,32005],
      "KEYS":["keyC","keyCD","keyAC","keyBC"]
    },
    "NOTHING":0
}
```

### 3PC-训练

> project: mpc-3pc

- 适用于三方有私有数据，进行模型训练的情况


|             |                            |
| ----------- | -------------------------- |
| 环境        | 4C-16G-100M/24C-64G-LAN    |
| 数据量      | 10w/20w/30w/50w/100w       |
| 特征数量    | 200                        |
| 数据划分⽅法 | 横向/纵向 （数据分为三份） |
| 模型        | 线性回归/逻辑回归          |
| 指标        | 时间/通信量/精度           |


**使用步骤：**

进入 `mpc-3pc/bin`。

```bash
cd bin
export LD_LIBRARY_PATH=../lib
```

**一，生成数据**

参考前文关于生成数据的详细说明。


<font style="color:red">

重要说明！这一步非常关键！
假设数据已经生成好了。其`根`路径为 `~/benchmark/`。
在当前目录`(bin/)`下建立一个data和conf的软链接，链接到对应路径。
比如，现在要进行 分类的训练，所选的样本数量为 1000。则

</font>

```bash
bin# rm data
bin# ln -sf ~/benchmark/cls/1000 data
bin# rm conf
bin# ln -sf ../conf conf
```


**二，配置文件**

参考上文关于配置文件的说明，结合生成的数据文件进行配置文件的配置。然后放到当前路径`(bin/)`的 `conf/` 下。

注意，每一个节点都需要一个配置文件，可以参考 `mpc-3pc/conf/` 下的默认配置。

重点注意 DATA_X/DATA_Y 是否配置正确。

如果是多机运行，重点注意 IP/PORT。


**三，开始训练**

数据已经生成好了，现在开始三方的训练。


1. 先看一下程序的帮助信息。

```
./mpc-3pc --help
usage: ./mpc-3pc --partyid=int --config_file=string [options] ...
options:
  -h, --help                  help (string [=this help list])
  -p, --partyid               0:P0;1:P1;2:P2 (int)
      --config_file           the config json file path for each party (string)
      --model                 model type(Logistic,Linear,CNN) (string [=Logistic])
      --is_train              0(false);1(true) (int [=1])
      --save_model_type       0(ciphertext);1(plaintext) (int [=0])
      --save_model_party      0:P0;1:P1. which party save the model (int [=0])
      --predict_recv_party    0:P0;1:P1. which party get the predictions (int [=0])
      --log_prefix            the log prefix path for cpp (string [=mpc])
      --model_file            the model stored path (string [=])

For example:
   ./mpc-3pc --partyid=0 --config_file=conf/homecredit0.json --model=Logistic
   ./mpc-3pc --partyid=1 --config_file=conf/homecredit1.json --model=Logistic
   ./mpc-3pc --partyid=2 --config_file=conf/homecredit2.json --model=Logistic
```

2. 准备好三个节点/终端。

此时当前路径`(bin/)`目录结构：

```
├── bench_data_gen.py # 生成数据用
├── classification_score.py # 分类模型评分用
├── conf/ # 配置文件目录
├── data -> ~/benchmark/cls/1000 # 数据目录
├── mpc-3pc # 可执行程序，用于 3pc 训练
└── regression_score.py # 回归模型评分用
```

- 节点/终端0 (P0)

```bash
mkdir -p log out
./mpc-3pc --partyid=0 --is_train=1 --log_prefix=log-3pc-train-Logistic \
    --save_model_type=1 --save_model_party=0 --model=Logistic \
    --config_file=conf/bench_train_v_0.json \
    --model_file=model-3pc-train-Logistic-0.model
```

- 节点/终端1 (P1)

```bash
mkdir -p log out
./mpc-3pc --partyid=1 --is_train=1 --log_prefix=log-3pc-train-Logistic \
    --save_model_type=1 --save_model_party=0 --model=Logistic \
    --config_file=conf/bench_train_v_1.json
```

- 节点/终端2 (P2)

```bash
mkdir -p log out
./mpc-3pc --partyid=2 --is_train=1 --log_prefix=log-3pc-train-Logistic \
    --save_model_type=1 --save_model_party=0 --model=Logistic \
    --config_file=conf/bench_train_v_2.json
```

参数说明：

| 参数               | 描述                                     |
| ------------------ | ---------------------------------------- |
| `partyid`          | 指定当前是哪一方。                       |
| `is_train`         | 是否是训练，三方要一样。                 |
| `log_prefix`       | 日志前缀。                               |
| `save_model_type`  | 模型保存形式，三方要一样。               |
| `save_model_party` | 模型保存在哪一方，三方要一样。           |
| `model`            | 当前的模型，三方要一样。                 |
| `config_file`      | 配置文件，三方要**不**一样。             |
| `model_file`       | 模型输出文件，与save_model_party要一致。 |

注意：

`is_train/save_model_type/save_model_party/model` 三方要一样。

`model_file` 训练时(`--is_train=1`)是输出，预测时(`--is_train=0`)是输入。

**模型输出**

模型输出，为了方便下面进行（⼀⽅模型，⼀⽅验证数据集）预测，将模型结果保存为明文，存储在当前路径 `out/` 下。具体路径由 `--model_file` 指定。


### 2PC-训练

> project: mpc-2pc


|             |                           |
| ----------- | ------------------------- |
| 环境        | 4C-16G-100M/24C-64G-LAN   |
| 数据量      | 10w/20w/30w/50w/100w      |
| 特征数量    | 200                       |
| 数据划分⽅法 | 横向/纵向（数据分为两份） |
| 模型        | 线性回归/逻辑回归         |
| 指标        | 时间/通信量/精度          |



```
root@vm:~/x/check/mpc-2pc/bin# export LD_LIBRARY_PATH=../lib
root@vm:~/x/check/mpc-2pc/bin# ./mpc-2pc --help
usage: ./mpc-2pc --partyid=int --config_file=string [options] ...
options:
  -h, --help                  help (string [=this help list])
  -p, --partyid               0:P0;1:P1 (int)
      --config_file           the config json file path for each party (string)
      --model                 model type(Logistic,Linear,CNN) (string [=Logistic])
      --is_train              0(false);1(true) (int [=1])
      --save_model_type       0(ciphertext);1(plaintext) (int [=0])
      --save_model_party      0:P0;1:P1. which party save the model (int [=0])
      --predict_recv_party    0:P0;1:P1. which party get the predictions (int [=0])
      --log_prefix            the log prefix path for cpp (string [=mpc])

For example:
   ./mpc-2pc --partyid=0 --config_file=conf/homecredit0.json --model=Logistic
   ./mpc-2pc --partyid=1 --config_file=conf/homecredit1.json --model=Logistic
```

使用方式参考 [3PC-训练]。

注意：2PC 除了少一个节点外，其它与 3PC 一致。


### 2PC-预测 I

> project: mpc-2pc


|             |                         |
| ----------- | ----------------------- |
| 环境        | 4C-16G-100M/24C-64G-LAN |
| 数据量      | 10w/20w/30w/50w/100w    |
| 特征数量    | 200                     |
| 数据划分⽅法 | ⼀⽅模型，⼀⽅验证数据集    |
| 模型        | 线性回归/逻辑回归       |
| 指标        | 时间/通信量/精度        |


```
root@vm:~/x/check/mpc-2pc/bin# export LD_LIBRARY_PATH=../lib
root@vm:~/x/check/mpc-2pc/bin# ./mpc-2pc --help
usage: ./mpc-2pc --partyid=int --config_file=string [options] ...
options:
  -h, --help                  help (string [=this help list])
  -p, --partyid               0:P0;1:P1 (int)
      --config_file           the config json file path for each party (string)
      --model                 model type(Logistic,Linear,CNN) (string [=Logistic])
      --is_train              0(false);1(true) (int [=1])
      --save_model_type       0(ciphertext);1(plaintext) (int [=0])
      --save_model_party      0:P0;1:P1. which party save the model (int [=0])
      --predict_recv_party    0:P0;1:P1. which party get the predictions (int [=0])
      --log_prefix            the log prefix path for cpp (string [=mpc])
      --model_file            the model stored path (string [=])

For example:
   ./mpc-2pc --partyid=0 --config_file=conf/homecredit0.json --model=Logistic
   ./mpc-2pc --partyid=1 --config_file=conf/homecredit1.json --model=Logistic
```


使用方式参考 [2PC/3PC-训练]。

**注意：**

默认，P0 拥有模型（即上面模型训练 `--model_file` 所指定的文件）；P1 拥有数据。

预测的结果输出可以由 `--predict_recv_party` 指定。

默认情况下，预测结果会输出到由 `--predict_recv_party` 指定节点的 `log/xxxx-predictY.csv` 下。


**评分**

在 `data/` 目录有一个 `pred_Y_target.csv` 文件（用于评分，可参考生成数据部分）。为简化处理，假设文件路径为 `/path/to/realY.csv`。

在 `log/` 目录有一个由上一步预测得到的预测值文件。为简化处理，假设文件名为 `/path/to/predY.csv`。


下面使用预测得到的结果与样本真实值进行评分。

- 逻辑回归评分

```bash
python classification_score.py /path/to/realY.csv /path/to/predY.csv
```

输出如下结果

```
{'score_auc': 0.9875225073899406, 'score_ks': 0.907267448994448, 'threshold_opt': 0.540771484375, 'score_accuracy': 0.953635, 'score_precision': 0.9587033478304845, 'score_recall': 0.9480880602538584, 'score_f1': 0.9533661558886983}
```

- 线性回归评分

```bash
python regression_score.py /path/to/realY.csv /path/to/predY.csv
```

输出如下结果

```
{'r2_score': 0.9998390245233595}
```


### 2PC-预测 II

> project: gbdt

|             |                           |
| ----------- | ------------------------- |
| 环境        | 4C-16G-100M / 24C-64G-LAN |
| 数据量(树)  | 10颗/50颗/100颗/500颗     |
| 特征数量    | 4                         |
| 数据划分⽅法 | ⼀⽅模型，⼀⽅验证数据集      |
| 模型        | GBDT                      |
| 指标        | 时间/通信量/精度          |


  
使用 （两个节点/终端）

- 进入 gbdt/bin

```bash
cd ./gbdt/bin
```

- 拷贝 上层 data 到当前目录

```bash
cp -rf ../data ./
```

- 显示 `data/` 目录结构

```bash
tree ./data
```

```
data/
├── data // 数据
│   ├── data0.txt
│   ├── ...
│   └── data9.txt
├── meta.txt
├── model // 模型
│   └── GBDT_scan.txt
├── score.py // 评分脚本
└── y_real.csv // 真实值
```


- 进行预测前，1. 导入动态库路径

```bash
export LD_LIBRARY_PATH=../lib
```


- 进行预测前，2. 先看一下命令行选项

```bash
./gbdt --help
```
```
usage: ./gbdt --partyid=int [options] ...
options:
  -h, --help        help (string [=this help list])
  -p, --partyid     1:ALICE;2:BOB (int)
      --data_dir    data path (string [=./data])
      --res_file    result path (string [=./res_file.csv])
      --host        server(alice) host/ip (string [=127.0.0.1])
      --port        server(alice) port (int [=11121])
      --loops       how many loops will be executed, for bench test (int [=1])

For example:
   ./gbdt --partyid=1 --data_dir=./data --port=11121
   ./gbdt --partyid=2 --data_dir=./data --port=11121
```

说明：ALICE 是模型拥有方，BOB 是数据拥有方。

- 进行预测，节点/终端 1 (ALICE)

```bash
./gbdt --partyid=1 --data_dir=./data/model \
   --host=127.0.0.1 --port=11121
```

- 进行预测，节点/终端 2 (BOB)

```bash
./gbdt --partyid=2 --data_dir=./data/data  \
   --host=127.0.0.1 --port=11121 \
   --res_file="data/y_pred.csv"
```

| 参数       | 描述                                   |
| ---------- | -------------------------------------- |
| `partyid`  | 指定当前是哪一方。                     |
| `data_dir` | ALICE 指定模型路径，BOB 指定数据路径。 |
| `host`     | ALICE 的 IP。                          |
| `port`     | ALICE 的 PORT。                        |
| `res_file` | 预测结果路径，只对 BOB 有效。          |


预测结果会输出到 `--res_file` 指定的文件。(BOB 方)


- 进行评分。节点/终端 2 (BOB)


```bash
cd ./data; python score.py; cd ..
```

输出：

```
R2 y_real,y_pred: 0.8867038027519715
```

## 补充说明
### PSI 相关
DH-Balanced-Basic-PSI / DH-Unbalanced-Basic-PSI / DH-Unbalanced-Labeled-PSI 都在贵司机器 `172.18.164.135` 和 `172.18.164.136`上验证通过，您也可以按照本说明文档的具体步骤进行测试验证；

### 机器学习相关
2PC/3PC 机器学习相关的环境也在贵司机器 `172.18.164.135` 、`172.18.164.136` 和 `172.18.164.137` 上验证通过，其中  `172.18.164.135` 为 P0， `172.18.164.136` 为 P1，`172.18.164.137` 为 P2。

**注意：**
为了训练贵司提供的数据集，已经在机器 `172.18.164.135` 和 `172.18.164.136` 中配置好对应的数据集，两台机器存放数据集的路径和目录结构都是一致的，具体路径为：
```
~/platon/mpc-2pc/bin/benchmark/cls/50000-50d
~/platon/mpc-2pc/bin/benchmark/cls/50000-100d
~/platon/mpc-2pc/bin/benchmark/cls/50000-200d
~/platon/mpc-2pc/bin/benchmark/cls/100000-50d
~/platon/mpc-2pc/bin/benchmark/cls/100000-100d
~/platon/mpc-2pc/bin/benchmark/cls/100000-200d
~/platon/mpc-2pc/bin/benchmark/cls/200000-50d
~/platon/mpc-2pc/bin/benchmark/cls/200000-100d
~/platon/mpc-2pc/bin/benchmark/cls/200000-200d
```
后续如果要训练其他数据集可以按照这样的目录结构来创建新的目录，另外这些目录里面的具体文件名格式也都一样，如：
```
train_X_V_partyA.csv  //纵向训练 P0 的私有数据
train_X_V_partyB.csv  //纵向训练 P1 的私有数据
train_X_V_partyC.csv  //纵向训练 P2 的私有数据
train_X_H_partyA.csv  //横向训练 P0 的私有数据
train_X_H_partyB.csv  //横向训练 P1 的私有数据
train_X_H_partyC.csv  //横向训练 P2 的私有数据
pred_X.csv            //用来预测的 X 值，对于 P0 有模型，P1 有数据的场景，这个文件在 P1 端
pred_Y_target.csv     //用来测试的 X 值对应的标签值（Y 值），可用来评分，请参考前面章节
```

同样为了方便真实数据集在 2PC 场景下的训练和预测，已经在 `172.18.164.135` 和 `172.18.164.136` 这两台机器中配置好脚本，可以直接执行，这些脚本的名称和作用如下：
```
~/platon/mpc-2pc/bin/logistic_train_5w-50d.sh   //2PC 训练， P0 和 P1 都要执行该同样名称的脚本，执行没有先后顺序要求
~/platon/mpc-2pc/bin/logistic_train_5w-100d.sh
~/platon/mpc-2pc/bin/logistic_train_5w-200d.sh
~/platon/mpc-2pc/bin/logistic_train_10w-50d.sh
~/platon/mpc-2pc/bin/logistic_train_10w-100d.sh
~/platon/mpc-2pc/bin/logistic_train_10w-200d.sh
~/platon/mpc-2pc/bin/logistic_train_20w-50d.sh
~/platon/mpc-2pc/bin/logistic_train_20w-100d.sh
~/platon/mpc-2pc/bin/logistic_train_20w-200d.sh
~/platon/mpc-2pc/bin/logistic_pred.sh           //2PC 预测， P0（模型持有方，训练的时候如果没有指定模型输出路径默认在 out 目录下） 和 P1（预测数据持有方）都要执行该同样名称的脚本，执行没有先后顺序要求
```
