from hyperon import MeTTa
metta = MeTTa()
input_str = "((topic var AI) (topic var AI))"
atoms = metta.parse_all(input_str)
print(f"Type of atoms: {type(atoms)}")
print(f"Length of atoms: {len(atoms)}")
print(f"First atom: {atoms[0]}")
print(f"Type of first atom: {type(atoms[0])}")
try:
    children = atoms[0].get_children()
    print(f"Children: {children}")
    print(f"Type of children: {type(children)}")
    print(f"First child string: {str(children[0])}")
except Exception as e:
    print(f"Error getting children: {e}")
