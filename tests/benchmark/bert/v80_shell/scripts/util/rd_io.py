import xml.etree.ElementTree as ET
import sys

def extract_value(xml_file, key):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        namespace = {'spirit': 'http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009'}
        xpath_query = f".//spirit:value[@spirit:id='{key}']"
        element = root.find(xpath_query, namespace)

        if element is not None:
            print(element.text.strip())  # Output the value
        else:
            print(f"Key '{key}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_value.py <xml_file> <key>")
        sys.exit(1)

    xml_file = sys.argv[1]
    key = sys.argv[2]
    extract_value(xml_file, key)