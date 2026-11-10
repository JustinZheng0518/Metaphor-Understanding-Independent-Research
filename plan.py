太复杂了说简单一点
可以，简单成 4 步就行：

先把基础模型跑完整

Gao 2018：已完成
BiLSTM：已完成
BERT
RoBERTa
做最关键的比较

overall F1
seen verb F1
unseen verb F1
BERT/RoBERTa 比 Gao、BiLSTM 提升多少
加一个我们自己的小改进

先比较 CLS
target verb
CLS + target verb
如果有提升，就可以作为第一版方法点
写 arXiv v1
论文结构就是：

Introduction
Related Work
Dataset
Gao/BiLSTM/BERT/RoBERTa
Results
Seen vs unseen analysis
我们的小改进
Error Analysis
Conclusion
你现在最该做的其实只有一件事：

先把 BERT 跑通，然后 RoBERTa。

跑完这两个，我们再看结果决定“自己的方法”具体做什么，不需要现在把后面所有东西都设计死。

一句话路线：

Gao/BiLSTM → BERT → RoBERTa → unseen分析 → 小改进 → arXiv。