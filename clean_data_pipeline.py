import re
import pandas as pd

# clean data and store into parquet, without considering the data score for stackoverflow
file_path = './data/security.output.csv'
output_path = './data/security_stack_exchange.parquet'

df = pd.read_csv(file_path)
def clean_html_tags(text):
    cleaned_text = text.replace('<p>', '').replace('</p>', '')
    clean = re.compile('<.*?>')
    res = re.sub(clean, ' ', cleaned_text)
    return res

def clean_space(text):
    # Remove duplicate newline characters
    cleaned_text = re.sub(r'\n+', '\n', text)
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text.strip()  # Remove leading and trailing whitespace

# Step 1: Lowercase all text
df_cleaned = df.map(lambda x: x.lower() if isinstance(x, str) else x)


# Step 2: Clean HTML tags
df_cleaned['Question'] = df_cleaned['Question'].apply(clean_html_tags)
df_cleaned['Answer'] = df_cleaned['Answer'].apply(clean_html_tags)

# Step3: Clean duplicated \n and space
df_cleaned = df_cleaned.map(lambda x: clean_space(x) if isinstance(x, str) else x)


# Display the cleaned DataFrame
df_cleaned.sample(10)

def generate_training_data(df):
    data = []
    for index, row in df.iterrows():
        question = row['Question']
        answer = row['Answer']
        inst_template = "<s>[INST] {} [/INST] {} </s>"
        inst_qa = inst_template.format(question, answer)
        data.append(inst_qa)
    return data


training_data = generate_training_data(df_cleaned)
training_data_df = pd.DataFrame(training_data, columns=['train'])

# Save training data as a Parquet file without header
training_data_df.to_parquet(output_path, index=False)

output_df = pd.read_parquet(output_path)
print(output_df.head(10))