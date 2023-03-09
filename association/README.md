# Association Rule Mining

- rule-based ML algorithm
- insipred from former youtube (personal) recommendation algorithm
- see also: https://www.kaggle.com/code/mervetorkan/association-rules-with-python
- literature: https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
- libraries: mixend, PyCaret, apyori

# Association Rules (Apriori Alogirthm)
- study of correlations between variables e.g. for recommendations or shopping associations (cross marketing of beer and chips e.g.)
- apriori algorithm using Boolean Association Rules
- three parameters
    - support: probability of event to occur (number of transactions containing a product divided by all transactions)
    - confidence: measure of condit. probability, A occurs given B with the confidence X (number of transcations where both A and B occur divided by transcations with only A).
    - lift: increase in ratio of sale of B when A is sold (Confidence (A->B) divded by support of B)
- if lift=1 -> no assocation between products A and B (unlikely to be bought together)
- if lift=5 e.g. -> likelohood of buying A and B together is 5 times larger than likelihood of just buying B
