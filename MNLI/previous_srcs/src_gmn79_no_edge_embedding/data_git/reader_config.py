# MODEL NAME for not config execution
TRANSFORMER_NAME = "bert-base-uncased"

# param for reader
# ANLI
anli_p_field = "context"
anli_h_field = "hypothesis"
anli_l_field = "label"
anli_fields = [anli_p_field, anli_h_field, anli_l_field]
# MNLI, SNLI
default_fields = ["sentence1", "sentence2", "gold_label"]

# label
label_to_id = {
    "contradiction" : 0,
    "neutral" : 1,
    "entailment" : 2,
}
id_to_label = ["contradiction", "neutral", "entailment"]