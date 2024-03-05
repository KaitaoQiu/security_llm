import csv
import xml.sax
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class StackOverflowHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.questions_and_answers = []
        self.current_question = None
        self.current_answer = None
        self.in_question = False
        self.in_answer = False

    def startElement(self, name, attrs):
        if name == "row":
            post_type_id = attrs.get('PostTypeId')
            if post_type_id == '1':  # Question
                self.current_question = {
                    'question_id': attrs.get('Id'),
                    'title': attrs.get('Title'),
                    'question_body': attrs.get('Body'),
                    'tags': attrs.get('Tags'),
                    'answers': []
                }
                self.in_question = True
            elif post_type_id == '2' and self.in_question:  # Answer
                self.current_answer = attrs.get('Body')
                self.in_answer = True

    def endElement(self, name):
        if name == "row":
            if self.in_question:
                self.questions_and_answers.append(self.current_question)
                self.in_question = False
            elif self.in_answer:
                self.current_question['answers'].append(self.current_answer)
                self.in_answer = False

    def characters(self, content):
        pass

def extract_questions_and_answers(xml_file):
    handler = StackOverflowHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    with open(xml_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chunk_size = len(lines) // 3  # Divide the file into chunks for processing in parallel
    chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]

    with ThreadPoolExecutor(max_workers=10) as executor:
        for _ in tqdm(executor.map(parser.feed, chunks), total=len(chunks), desc="Processing XML"):
            pass

    return handler.questions_and_answers

def save_to_csv(questions_and_answers, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer', 'Tags'])
        for qa in questions_and_answers:
            question_title = qa['title']
            for answer in qa['answers']:
                writer.writerow([question_title, answer, qa['tags']])

# Example usage
xml_file_path = 'Posts.xml'
csv_file_path = 'output.csv'

questions_and_answers = extract_questions_and_answers(xml_file_path)
save_to_csv(questions_and_answers, csv_file_path)
