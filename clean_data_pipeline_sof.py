import re
import pandas as pd
import json

class StackOverFlowCleaner:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)

    def load_data(self):
        df = pd.read_csv(self.config['file_path'])
        print("finish loading data")
        return df

    # def load_data(self):
    #     chunksize = 10 ** 6  # Adjust chunksize to your needs
    #     total_size = sum(1 for row in open(self.config['file_path'], 'r', encoding='utf-8')) - 1
    #     progress_bar = tqdm(total=total_size, desc="Loading Data")
        
    #     # Initialize an empty DataFrame to hold chunks
    #     df = pd.DataFrame()

    #     # Process the CSV file in chunks
    #     for chunk in pd.read_csv(self.config['file_path'], chunksize=chunksize):
    #         df = pd.concat([df, chunk], ignore_index=True)
    #         progress_bar.update(chunksize)

    #     progress_bar.close()
    #     print("Finish loading data")
    #     return df

    def clean_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'<.*?>', ' ', text)  # Remove HTML tags
            text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces
        return text

    def filter_security_related(self, df):
        security_tags_df = pd.read_csv(self.config['security_tag_path'])
        security_tags_set = set(security_tags_df['Tag'])

        def is_security_related(tags):
            return any(tag.strip('<>') in security_tags_set for tag in tags.split('<'))

        df = df[df['Tags'].apply(is_security_related)]
        print("finish filter security tag")
        return df

    def filter_high_score(self, df):
        high_score_threshold = self.config['high_score_threshold']
        high_score_df = df[df['Score'] >= high_score_threshold]
        print("finish filter high score")
        return high_score_df
    
    # FIXME:
    def merge_duplicate_questions(self, df):
        df['Question'] = df['Question'].apply(self.clean_text)
        merged_df = df.groupby('Question')['Answer'].apply(' '.join).reset_index()
        print("merge duplicated questions")
        return merged_df

    def generate_training_data(self, df):
        training_data = []
        for _, row in df.iterrows():
            question = row['Question']
            answer = row['Answer']
            inst_template = "<s>[INST] {} [/INST] {} </s>"
            inst_qa = inst_template.format(question, answer)
            training_data.append(inst_qa)
        print("finish generating parquet traing data")
        return training_data

    def save_training_data(self, training_data):
        training_data_df = pd.DataFrame(training_data, columns=['train'])
        training_data_df.to_parquet(self.config['output_path'], index=False)


if __name__ == "__main__":
    config_file = './data_config/filter_sof_cleaner.json'
    data_processor = StackOverFlowCleaner(config_file)

    # Load data
    df = data_processor.load_data()

    # Clean and filter data
    df['Question'] = df['Question'].apply(data_processor.clean_text)
    df['Answer'] = df['Answer'].apply(data_processor.clean_text)
    df = data_processor.filter_security_related(df)
    high_score_df = data_processor.filter_high_score(df)

    # Merge duplicate questions and combine answers
    merged_df = crawler.merge_duplicate_questions(high_score_df)

    # Generate and save training data
    training_data = data_processor.generate_training_data(merged_df)
    data_processor.save_training_data(training_data)

    # Display the first 10 rows of the output
    output_df = pd.read_parquet(data_processor.config['output_path'])
    print(output_df.head(10))
