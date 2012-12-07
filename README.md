CS221 Final Project Documentation
### Author: 
<ul>
<li>Qingyi Meng (qingyim@stanford.edu)</li>
<li>Chang Su (changsu@stanford.edu)</li>
</ul>

### Methods
#### 1. Linear Regression

<b>Start the program</b>
<pre>run gradient_lr</pre>

<b>Result logging</b>
<pre>
MSE_Train =
MSE on training data using RL: 0.48753

MSE_Test =
MSE on testing data using RL: 0.81715

querySentencesPool = 
    '9.17.2'     [ 326]
    '7.11.4'     [ 628]
    '1.9.4'      [ 687]
    ...
    ...
</pre>
We output mean squared error (MSE) of linear regression on training data and testing data
seperately and also output pool of sentences that can be used for query. These sentences are the testing sentences that the model has never "seen" before
In addition, we also draw the learning curve of traning iterations vs mse similar to figure `learning_curve_lr.fig`

<b>Query</b>

After running the algorithm, we have generate a linear model ready for prediction. 
We can choose any sentence number in querySentencePool and issue the query using the cmd
<pre>query('9.17.2')</pre>
Then you will see result similar as below
<pre>
realValue =

    0.6829
    0.5371
    0.2839
    0.3478
    0.3376
    0.3529
    0.2506

estimatedValue =

    0.4005
    0.3774
    0.4736
    0.4021
    0.1091
    0.2080
    0.0759

result = 
    'norm2 distance: '    '0.5'    'Confidence: '    [391]

</pre>

Here we show ground truth value and predicted value and also their norm2 distance and our confidence in the prediction based on # of scripts generated from the sentence. 
In addition, we also visualize the result using bar graph.

#### 2. KL Optimization Method
<b>Star the program</b>
<pre>run gradient_kl</pre>

<b>Result Logging</b>
<pre>
Loss_KL =
Loss Function on testingn data using KL method: 4.4625

Loss_rand =
Loss Function on testingn data using random guess: 20.3272

querySentencesPool = 
    '1.4.5'      [  93]
    '9.18.4'     [ 636]
    '7.11.3'     [ 286]
    ...
    ...
</pre>

Here, we output Loss function on testing data using KL method and random guess respectively and also pool of sentences used for query
In addition, we also generate learning curves of the KL method iterations vs loss function in figure file `learning_curve_kl.fig`

<b>Query</b>

Similarly to the linear regression method, after running KL-method, we can also issue queries for prediction.
Then we will generate result similar to this:
<pre>
realValue =

    0.2101
    0.2502
    0.1366
    0.0721
    0.1709
    0.0866
    0.0735

estimatedValue =

    0.1572
    0.2248
    0.1084
    0.1059
    0.2012
    0.1187
    0.0837


result = 
    'KL distance: '    '0.027233'    'Confidence: '    [636]
</pre>


#### File Structures
`feature_matrix.txt` - 268 * 30 matrix with row as sentence column as features

`feature_name.txt` - 30 dimension vector that store name of each features

`label_dependent.txt` - 162 * 7 matrix with row as each sentence column as error type probabilities. For each row, all columns reflects the distribution of error types, thus summing up to 1

`label_independent.txt` - 162 * 7 matrix with row as each sentence, column as error type probabilities. However, it's differenct from `label_dependent.txt` file in a sense that there is no constraint among columns for a particular row and they are independent with each other. 

`sentence_map.txt` - store map between sentence number and total # of scripts genearted from that sentence

`sample_kl.fig` - query result figure of KL method

`sample_lr.fig` - query result figure of LR method		
		      			  





