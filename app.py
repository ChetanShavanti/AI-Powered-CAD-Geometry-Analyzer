import streamlit as st
import streamlit.components.v1 as components
import os
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False
from stl import mesh
from steputils import p21
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from io import BytesIO

def generate_explanation(file_format, geometry_type, data):
    """
    Generate a beginner-friendly AI-based explanation of the CAD file.
    This is a mock function; in production, integrate with an LLM API.
    """
    explanation = f"**What is this format?**\n"
    explanation += f"This is a {file_format} file, which represents {geometry_type} geometry. "

    if geometry_type == "B-Rep":
        if 'faces' in data:
            explanation += f"It contains {data['faces']} faces and {data['edges']} edges. "
        else:
            explanation += f"It has {data['entities']} entities. "
        explanation += "B-Rep (Boundary Representation) uses exact mathematical surfaces for precise modeling.\n\n"
        explanation += "**When should you use it?**\n"
        explanation += "Best for engineering designs, simulations, and manufacturing where accuracy is critical.\n\n"
        explanation += "**Risks:**\n"
        explanation += "Larger file sizes and requires specialized software for editing."
    elif geometry_type == "Mesh":
        explanation += f"It has {data['triangles']} triangles. "
        bbox = data['bbox']
        explanation += f"Bounding box: X: {bbox['x'][0]:.2f} to {bbox['x'][1]:.2f}, Y: {bbox['y'][0]:.2f} to {bbox['y'][1]:.2f}, Z: {bbox['z'][0]:.2f} to {bbox['z'][1]:.2f} (units assumed mm).\n\n"
        explanation += "**When should you use it?**\n"
        explanation += "Ideal for 3D printing, gaming, animations, and rapid prototyping.\n\n"
        explanation += "**Risks:**\n"
        explanation += "Approximate geometry can lead to accuracy loss, visible faceting, and is not suitable for parametric edits or high-precision simulations."

    return explanation

def detect_format(file_path):
    """
    Detect the file format based on extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.step', '.stp']:
        return 'STEP'
    elif ext == '.stl':
        return 'STL'
    else:
        return None

def analyze_step(file_path):
    """
    Analyze STEP file using steputils to validate and count entities.
    If OCC available, use it for detailed analysis.
    """
    if OCC_AVAILABLE:
        reader = STEPControl_Reader()
        status = reader.ReadFile(file_path)
        if status != 1:
            raise ValueError("Failed to read STEP file")
        reader.TransferRoots()
        shape = reader.OneShape()

        # Count faces
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = 0
        while explorer.More():
            faces += 1
            explorer.Next()

        # Count edges
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        edges = 0
        while explorer.More():
            edges += 1
            explorer.Next()

        return {'faces': faces, 'edges': edges}
    else:
        # Use steputils for basic parsing
        try:
            with open(file_path, 'r') as f:
                data = p21.load(f)
            entities = len(data)
            return {'entities': entities}
        except Exception as e:
            raise ValueError(f"Failed to parse STEP file: {e}")

def analyze_stl(file_path):
    """
    Analyze STL file using numpy-stl to extract triangle count and bounding box.
    """
    stl_mesh = mesh.Mesh.from_file(file_path)
    triangles = stl_mesh.data.shape[0]
    min_coords = stl_mesh.min_
    max_coords = stl_mesh.max_
    bbox = {
        'x': [float(min_coords[0]), float(max_coords[0])],
        'y': [float(min_coords[1]), float(max_coords[1])],
        'z': [float(min_coords[2]), float(max_coords[2])]
    }
    return {'triangles': triangles, 'bbox': bbox, 'mesh': stl_mesh}

# Streamlit Web App
st.title("🤖 AI-Powered CAD Geometry Analyzer")
st.markdown("### Hackathon POC: Understand STEP & STL Files with AI Explanations")
st.write("Upload a CAD file (.step, .stp, or .stl) to analyze its geometry and get an AI-powered explanation.")

# Add static info section
with st.expander("ℹ️ Learn About CAD Formats"):
    st.markdown("""
    **B-Rep (Boundary Representation)**: Used in STEP files. Represents geometry with exact surfaces (planes, cylinders, etc.) for precise modeling.
    - **Advantages**: High accuracy, parametric editing.
    - **Use Cases**: Engineering, simulations, manufacturing.
    - **Limitations**: Larger files, requires specialized software.

    **Mesh (Triangle Mesh)**: Used in STL files. Approximates geometry with triangles.
    - **Advantages**: Simple, good for visualization and 3D printing.
    - **Use Cases**: 3D printing, gaming, animations.
    - **Limitations**: Approximate (faceting), not ideal for precise edits or simulations.
    """)

uploaded_file = st.file_uploader("Choose a CAD file", type=['step', 'stp', 'stl'], help="Supported formats: STEP (.step/.stp) for B-Rep, STL (.stl) for Mesh")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📁 File Info")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024 / 1024:.2f} MB")

    # Save uploaded file to a temporary location
    temp_file_path = "temp_cad_file"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Detect format
    file_format = detect_format(uploaded_file.name)
    if file_format is None:
        st.error("❌ Unsupported file format. Please upload .step, .stp, or .stl files.")
    else:
        with col2:
            st.success(f"✅ Detected {file_format} file.")

        data = None
        geometry_type = None

        try:
            if file_format == 'STEP':
                geometry_type = "B-Rep"
                data = analyze_step(temp_file_path)
                with st.container():
                    st.subheader("🔍 Geometry Data (B-Rep)")
                    if 'faces' in data:
                        st.metric("Faces", data['faces'])
                        st.metric("Edges", data['edges'])
                    else:
                        st.metric("Entities", data['entities'])
                        st.info("💡 Detailed face/edge counts require pythonOCC (install via conda for full analysis).")

                    # Preview note
                    st.subheader("👁️ 3D Preview")
                    st.info("3D preview for STEP files requires advanced CAD libraries. For this POC, preview is not available.")

            elif file_format == 'STL':
                geometry_type = "Mesh"
                data = analyze_stl(temp_file_path)
                with st.container():
                    st.subheader("🔍 Geometry Data (Triangle Mesh)")
                    st.metric("Triangles", data['triangles'])
                    bbox = data['bbox']
                    st.write(f"**Bounding Box (mm):**")
                    st.write(f"X: {bbox['x'][0]:.2f} - {bbox['x'][1]:.2f}")
                    st.write(f"Y: {bbox['y'][0]:.2f} - {bbox['y'][1]:.2f}")
                    st.write(f"Z: {bbox['z'][0]:.2f} - {bbox['z'][1]:.2f}")
                    st.info("📐 STL uses triangles to approximate surfaces. Each triangle is a flat face connecting 3 vertices.")

                    # Generate and display 3D preview
                    st.subheader("👁️ 3D Preview")
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(data['mesh'].vectors, facecolors='cyan', linewidths=0.1, edgecolors='k', alpha=0.5))
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title('STL Mesh Preview')
                    ax.set_xlim(bbox['x'])
                    ax.set_ylim(bbox['y'])
                    ax.set_zlim(bbox['z'])
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    st.image(buf, caption="3D Mesh Preview", use_column_width=True)
                    plt.close(fig)

            # Generate and display AI explanation
            if data:
                st.subheader("🤖 AI Explanation")
                explanation = generate_explanation(file_format, geometry_type, data)
                st.markdown(explanation)

        except Exception as e:
            st.error(f"❌ Error analyzing the file: {str(e)}")

    # Clean up temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)