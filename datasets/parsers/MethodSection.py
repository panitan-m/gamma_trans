class MethodSection():
    def __init__(self, PAPER):
        self.ID = PAPER.ID
        self.TITLE = PAPER.TITLE
        self.METHOD = None

    @staticmethod
    def from_paper(paper):
        method_section = MethodSection(paper)

        sections = paper.SCIENCEPARSE.get_sections_dict()
        section_num = -1
        passed_exp = False
        for section_title, section_content in sections.items():
            added = False
            if section_num == section_title[0]:
                added = True

            for section_key in ['Introduction', 'Related work', 'Experiment', 'Conclusion']:
                if section_key.lower() in section_title.lower():
                    section_num = section_title[0]
                    added = True
                    if 'experiment' in section_title.lower():
                        passed_exp = True
            
            if not added and not passed_exp and section_title != 'None':
                section = {
                    "heading": section_title,
                    "text": section_content,
                    "is_method": 1
                }
                if method_section.METHOD is None:
                    method_section.METHOD = [section]
                else:
                    method_section.METHOD.append(section)
        return method_section
