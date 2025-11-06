# ML_LinearModule
编写Python代码，通过complete_train_samples.csv搭建、训练线性回归模型和KNN模型，利用XA, XB预测Y。
用matplotlib画出针对complete_train_samples中样本的模型预测值和Y的真实值。
计算模型在complete_train_samples的预测表现(R^2)。
预测test_samples.csv中的各样本的Y值，并把结果保存在’test_prediction.csv’中（含有XA, XB, Prediction三列）。

我们将按照以下步骤进行：
步骤1：导入必要的库
步骤2：输出姓名和问题答案
步骤3：加载训练数据complete_train_samples.csv
步骤4：数据预处理（如果有需要）
步骤5：训练线性回归模型和KNN模型
步骤6：在训练集上预测并计算R^2
步骤7：绘制真实值与预测值的散点图（两个模型分别绘制）
步骤8：加载测试数据test_samples.csv，用训练好的模型进行预测
步骤9：将测试集的预测结果保存为test_prediction.csv
