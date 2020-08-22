TRANSFORMER_NAME = "bert-base-uncased"


# alias and for changing, on for in(raw), one for saved(parsed)
# o_not used yet
anli_p_field = "context"
anli_h_field = "hypothesis"
anli_l_field = "label"
anli_fields = [anli_p_field, anli_h_field, anli_l_field]


default_fields = ["sentence1", "sentence2", "gold_label"]

label_to_id = {
    "contradiction" : 0,
    "neutral" : 1,
    "entailment" : 2,
}
id_to_label = ["contradiction", "neutral", "entailment"]