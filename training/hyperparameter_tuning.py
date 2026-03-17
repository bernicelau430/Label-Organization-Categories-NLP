from knn_model_v3 import main

word_grams = [(1,1), (1,2), (1,3)]
char_grams = [(2,5), (3,5), (2,6), (3,6)]
sublinear_tfs = [True, False]

results = []

for k in range(3, 11, 2):
    for word_gram in word_grams:
        for char_gram in char_grams:
            for conf in range(0.35, 0.55, 0.02):
                res = main(k, word_gram, char_gram, conf)
                results.append({
                    "accuracy": res["overall_accuracy"],
                    "params": [k, word_gram, char_gram, conf]
                })

results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

print(results[:100])