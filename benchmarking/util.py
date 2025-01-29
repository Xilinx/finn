# Utility functions for benchmarking
import os, shutil
from qonnx.core.datatype import DataType
import xml.etree.ElementTree as ET

def _find_rows_and_headers(table):
    rows = table.findall("tablerow")
    headers = []

    for row in rows:
        headers = row.findall("tableheader")
        if len(headers) > 0:
            break
    return (rows, headers)


def summarize_table(table):
    table_summary = {}
    table_summary["headers"] = []
    rows, headers = _find_rows_and_headers(table)

    if len(headers) > 0:
        string = "Header: "
        for header in headers:
            table_summary["headers"].append(header.attrib["contents"])
            string = string + header.attrib["contents"] + " "
        # print(string.rstrip())

    for row in rows:
        cells = row.findall("tablecell")
        if len(cells) > 0:
            cell_name = cells[0].attrib["contents"]
            string = cell_name
            table_summary[cell_name] = []
            for cell in cells[1:]:
                table_summary[cell_name].append(cell.attrib["contents"])
                string = string + cell.attrib["contents"] + " "
            # print(string.rstrip())

    return table_summary


def summarize_section(section):
    section_summary = {}
    section_summary["tables"] = []
    section_summary["subsections"] = {}

    # print("Section:", section.attrib["title"])
    tables = section.findall("table")
    sub_sections = section.findall("section")
    for table in tables:
        section_summary["tables"].append(summarize_table(table))
    # print("")
    for sub_section in sub_sections:
        section_summary["subsections"][sub_section.attrib["title"]] = summarize_section(sub_section)

    return section_summary


def power_xml_to_dict(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sections = root.findall("section")
    result = {}

    for section in sections:
        result[section.attrib["title"]] = summarize_section(section)

    return result

def prepare_inputs(input_tensor, idt, wdt):
    if wdt == DataType["BIPOLAR"] and idt == DataType["BIPOLAR"]:
        # convert bipolar to binary
        return {"inp": (input_tensor + 1) / 2}
    else:
        return {"inp": input_tensor}

def delete_dir_contents(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
