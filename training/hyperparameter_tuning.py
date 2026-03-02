from model_training import main

word_grams = [(1,1), (1,2), (1,3)]
min_dfs = [1,2,3]
max_dfs = [0.8, 0.9, 1.0]
sublinear_tfs = [True, False]

max_accuracy = 0
max_hyperparams = []

for k in range(3, 13, 2):
    for word_gram in word_grams:
        for min_df in min_dfs:
            for max_df in max_dfs:
                for sub in sublinear_tfs:
                    res = main(k, "word", word_gram, min_df, max_df, sub)
                    if res['overall_accuracy'] > max_accuracy:
                        max_accuracy = res['overall_accuracy']
                        max_hyperparams = [k, "word", word_gram, min_df, max_df, sub]

char_grams = [(2,5), (3,5), (2,6), (3,6), (2,7), (3,7)]

for k in range(3, 13, 2):
    for char_gram in char_grams:
        for min_df in min_dfs:
            for max_df in max_dfs:
                for sub in sublinear_tfs:
                    res = main(k, "char_wb", word_gram, min_df, max_df, sub)
                    if res['overall_accuracy'] > max_accuracy:
                        max_accuracy = res['overall_accuracy']
                        max_hyperparams = [k, "char_wb", char_gram, min_df, max_df, sub]

print(max_accuracy)
print(max_hyperparams)

# result: [3, 'word', (1, 1), 1, 0.8, True]