# must print pipeline objects after calling object
from transformers import pipeline

# Sentiment-analysis: good/bad
classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))
print(classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I dislike emch and i would like to kill myself!!"]
))

# Zero-shot classification: like above but with any labels
classifier = pipeline("zero-shot-classification")
print(classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
))

# Text generation: complete the sentence
generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to"))

# Mask filling: the model fills the special mask token
unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you all about <mask> models.", top_k=2))

# Named entity recognition: can label proper nouns such as people or places
ner = pipeline("ner", grouped_entities=True)
print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

# Question answering: extracts information from the provided context and does not generate an answer
question_answerer = pipeline("question-answering")
print(question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
))

# Summarization: reduces text while keeping important aspects
summarizer = pipeline("summarization")
print(summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
))

# requires sentencepiece to be installed
# Translation: translates from one language to another (recommended to use hub models)
# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# print(translator("Ce cours est produit par Hugging Face."))
