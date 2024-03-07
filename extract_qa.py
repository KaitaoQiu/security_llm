import csv
import xml.sax
from tqdm import tqdm

class StackOverflowHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.questions_and_answers = []
        self.current_question = None
        self.current_answer = None

    def startElement(self, name, attrs):
        if name == "row":
            post_type_id = attrs.get('PostTypeId')
            if post_type_id == '1':
                self.current_question = {
                    'title': attrs.get('Title'),
                    'question_body': attrs.get('Body'),
                    'tags': attrs.get('Tags'),
                    'score': attrs.get('Score'),
                    'favorite_count': attrs.get('FavoriteCount'),
                    'view_count': attrs.get('ViewCount')
                }
            elif post_type_id == '2':
                self.current_answer = attrs.get('Body')

    def endElement(self, name):
        if name == "row":
            if self.current_question is not None and self.current_answer is not None:
                self.questions_and_answers.append((
                    self.current_question['title'],
                    self.current_answer,
                    self.current_question['tags'],
                    self.current_question['score'],
                    self.current_question['favorite_count'],
                    self.current_question['view_count']
                ))

def extract_questions_and_answers(xml_file, max_size=None):
    handler = StackOverflowHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)

    current_size = 0
    with open(xml_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Processing XML"):
            current_size += len(line.encode('utf-8'))  # Calculate size of the line in bytes
            if max_size is not None:
                if current_size >= max_size:
                    break
            parser.feed(line)

    return handler.questions_and_answers

def save_to_csv(questions_and_answers, csv_file):
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer', 'Tags', 'Score', 'Favorite Count', 'View Count'])
        for qa in questions_and_answers:
            writer.writerow(qa)

# Example usage
xml_file_path = './stackoverflow/Posts.xml'
csv_file_path = './data/output.csv'
# max_xml_size = 1 * 1024 * 1024 * 1024  # GB in bytes
max_xml_size = None

questions_and_answers = extract_questions_and_answers(xml_file_path, max_xml_size)
save_to_csv(questions_and_answers, csv_file_path)
