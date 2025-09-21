from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

input_text = """
    The quick brown fox jumps over the lazy dog. The sentence is an example of a pangram, which is a sentence containing every letter of the alphabet at least once. While it is used to test fonts and typewriters, the sentence has become a part of typographic history and is considered an important phrase for learning the alphabet. It was first used in 1885 by a typewriter salesman and has been used in various contexts since. For instance, it is sometimes used by typists to practice typing speed and accuracy. Furthermore, it is sometimes used in the testing of keyboard layouts or text-processing systems.
    """

def split_text(text, chunk_size=1000):
    """
    Splits a large text into smaller chunks based on token limit
    """
    # Split the text into smaller chunks based on chunk_size
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

chunks = split_text(input_text, chunk_size=200)  # 200 words per chunk for example

# Summarize each chunk
summaries = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]

# Combine all summaries into one
full_summary = " ".join(summaries)
print(full_summary)


# optional- to get the summary in text file
# Save the summary to a text file
## summary_text = summary[0]['summary_text']

## with open("/content/summary.txt", "w") as f:
##    f.write(summary_text)

# Provide a link to download the summary
## from google.colab import files
## files.download("/content/summary.txt")
