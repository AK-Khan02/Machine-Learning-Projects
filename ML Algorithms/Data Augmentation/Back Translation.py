from transformers import MarianMTModel, MarianTokenizer

def back_translate(sentence, src_lang='en', mid_lang='fr'):
    """
    Translate a sentence to 'mid_lang' and back to 'src_lang'.
    """
    # Load models and tokenizers for both translation directions
    src_to_mid_model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{mid_lang}'
    mid_to_src_model_name = f'Helsinki-NLP/opus-mt-{mid_lang}-{src_lang}'

    src_to_mid_model = MarianMTModel.from_pretrained(src_to_mid_model_name)
    src_to_mid_tokenizer = MarianTokenizer.from_pretrained(src_to_mid_model_name)

    mid_to_src_model = MarianMTModel.from_pretrained(mid_to_src_model_name)
    mid_to_src_tokenizer = MarianTokenizer.from_pretrained(mid_to_src_model_name)

    # Translate to mid language
    mid_lang_tokens = src_to_mid_tokenizer(sentence, return_tensors='pt', padding=True)
    mid_lang_text = src_to_mid_model.generate(**mid_lang_tokens)
    mid_sentence = src_to_mid_tokenizer.decode(mid_lang_text[0], skip_special_tokens=True)

    # Translate back to source language
    src_lang_tokens = mid_to_src_tokenizer(mid_sentence, return_tensors='pt', padding=True)
    src_lang_text = mid_to_src_model.generate(**src_lang_tokens)
    back_translated_sentence = mid_to_src_tokenizer.decode(src_lang_text[0], skip_special_tokens=True)

    return back_translated_sentence
