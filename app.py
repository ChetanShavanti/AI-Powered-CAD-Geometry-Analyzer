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
import requests

def generate_explanation(file_format, geometry_type, data):
    """
    Generate a comprehensive AI-based explanation using Hugging Face Inference Providers API.
    """
    api_token = "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"  # API token hardcoded as requested

    # 1. Format the statistics for the prompt
    stats_str = ""
    if file_format == 'STL':
        bbox = data.get('bbox', {})
        dims = [
            bbox['x'][1] - bbox['x'][0] if 'x' in bbox else 0,
            bbox['y'][1] - bbox['y'][0] if 'y' in bbox else 0,
            bbox['z'][1] - bbox['z'][0] if 'z' in bbox else 0
        ]
        stats_str = (
            f"- **Triangle Count:** {data.get('triangles', 'N/A')}\n"
            f"- **Bounding Box Dimensions (mm):** {dims[0]:.2f} x {dims[1]:.2f} x {dims[2]:.2f}"
        )
    elif file_format == 'STEP':
        if 'faces' in data:
            stats_str = (
                f"- **Face Count:** {data.get('faces', 'N/A')}\n"
                f"- **Edge Count:** {data.get('edges', 'N/A')}"
            )
        else:
            stats_str = f"- **Entity Count:** {data.get('entities', 'N/A')}"

    # 2. Create the new, detailed prompt
    prompt = f"""
You are GeoMind AI, an expert assistant specializing in CAD geometry analysis.

You have analyzed a CAD file with the following properties:
- **File Format:** {file_format}
- **Geometry Type:** {geometry_type}
- **Key Statistics:**
{stats_str}

Based on this information, provide a detailed analysis for the user. Structure your response in the following sections using Markdown:

### 🤖 AI-Powered Analysis

**1. Geometry Overview:**
Briefly describe what these statistics mean in practical terms (e.g., is the triangle count high or low? What does the bounding box tell us about the object's scale?).

**2. Potential Use Cases:**
Based on the format and complexity, list 3-5 specific and practical use cases. Be creative. For example, instead of just "3D printing," suggest "a housing for a Raspberry Pi project" or "a scale model for an architectural presentation."

**3. Design & Manufacturing Insights:**
- For **STL (Mesh)**: Discuss potential issues like mesh integrity (watertightness), non-manifold geometry, and how the triangle count (faceting) could impact the surface finish. Suggest software (like Blender or Meshmixer) for validation or repair.
- For **STEP (B-Rep)**: Discuss the benefits of precise geometry for tasks like CNC machining, injection molding, or Finite Element Analysis (FEA). Mention the importance of model quality and continuity between surfaces.

**4. Actionable Recommendations:**
Provide a bulleted list of 2-3 concrete next steps for the user.
- **Example for STL:** "Recommendation: Run a mesh analysis in a tool like Meshmixer to check for print-killing errors before sending to a 3D printer."
- **Example for STEP:** "Recommendation: Import this model into a CAD assembly to verify its fit and interfaces with other parts."
"""

    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    data_payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,  # Increased for a more detailed response
        "temperature": 0.7
    }

    try:
        resp = requests.post("https://router.huggingface.co/v1/chat/completions", headers=headers, json=data_payload)
        if resp.status_code == 200:
            result = resp.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"]
            else:
                st.warning(f"API returned an unexpected format: {result}")
                return "AI explanation could not be fully generated. The model's response was not in the expected format."
        else:
            st.error(f"API Error: {resp.status_code} - {resp.text}")
            return f"Failed to get AI explanation. The model may be unavailable or the request may have failed."
    except Exception as e:
        return f"Hugging Face API Error: {e}. Please check your connection and API token."

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

# Add static info section
st.title("CAD Decoded / GeoMind AI - 🤖 AI-Powered CAD Geometry Analyzer")
st.markdown("### Hackathon POC: Understand STEP & STL Files with AI Explanations")
st.write("👋 Welcome! Upload a CAD file to analyze its geometry, see a 3D preview (for STL), and get AI-powered insights on use cases, accuracy, and impacts in design, manufacturing, and 3D printing.")

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
                    with st.spinner("Generating 3D preview..."):
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
                        st.image(buf, width=600, caption="3D Mesh Preview")
                        plt.close(fig)

            # Classification and Summary
            st.subheader("📊 Classification & Summary")
            col_class, col_summary = st.columns(2)
            with col_class:
                st.write(f"**Geometry Type:** {geometry_type}")
                if geometry_type == "B-Rep":
                    st.success("✅ Precise, parametric representation")
                else:
                    st.success("✅ Approximate, triangulated representation")
            with col_summary:
                if geometry_type == "B-Rep":
                    summary = f"B-Rep geometry with {'faces and edges' if 'faces' in data else 'entities'} for exact modeling."
                else:
                    summary = f"Triangle mesh with {data['triangles']} triangles, bounding box dimensions shown above."
                st.write(f"**Quick Summary:** {summary}")

            # Insight Cards
            st.subheader("💡 Key Insights")
            insight_cols = st.columns(3)
            with insight_cols[0]:
                st.markdown(f"""
                <div style="border: 1px solid #666; border-radius: 10px; padding: 10px; background: linear-gradient(to bottom, #333333, #555555); color: white; text-align: center;">
                    <h4>Geometry Type</h4>
                    <p>{geometry_type}</p>
                </div>
                """, unsafe_allow_html=True)
            with insight_cols[1]:
                st.markdown(f"""
                <div style="border: 1px solid #666; border-radius: 10px; padding: 10px; background: linear-gradient(to bottom, #333333, #555555); color: white; text-align: center;">
                    <h4>Best Use Case</h4>
                    <p>{"Engineering & Simulations" if geometry_type == "B-Rep" else "3D Printing & Visualization"}</p>
                </div>
                """, unsafe_allow_html=True)
            with insight_cols[2]:
                st.markdown(f"""
                <div style="border: 1px solid #666; border-radius: 10px; padding: 10px; background: linear-gradient(to bottom, #333333, #555555); color: white; text-align: center;">
                    <h4>Accuracy Level</h4>
                    <p>{"High (Exact)" if geometry_type == "B-Rep" else "Approximate (Faceted)"}</p>
                </div>
                """, unsafe_allow_html=True)

            # Additional Insight Cards
            st.subheader("🔍 More Insights")
            more_cols = st.columns(3)
            with more_cols[0]:
                st.markdown(f"""
                <div style="border: 1px solid #666; border-radius: 10px; padding: 10px; background: linear-gradient(to bottom, #333333, #555555); color: white; text-align: center;">
                    <h4>File Size Impact</h4>
                    <p>{"Larger files (complex)" if geometry_type == "B-Rep" else "Smaller files (efficient)"}</p>
                </div>
                """, unsafe_allow_html=True)
            with more_cols[1]:
                st.markdown(f"""
                <div style="border: 1px solid #666; border-radius: 10px; padding: 10px; background: linear-gradient(to bottom, #333333, #555555); color: white; text-align: center;">
                    <h4>Editability</h4>
                    <p>{"Parametric (easy edits)" if geometry_type == "B-Rep" else "Limited (mesh fixes needed)"}</p>
                </div>
                """, unsafe_allow_html=True)
            with more_cols[2]:
                st.markdown(f"""
                <div style="border: 1px solid #666; border-radius: 10px; padding: 10px; background: linear-gradient(to bottom, #333333, #555555); color: white; text-align: center;">
                    <h4>Processing Speed</h4>
                    <p>{"Slower (detailed)" if geometry_type == "B-Rep" else "Faster (simple)"}</p>
                </div>
                """, unsafe_allow_html=True)

            # Generate and display AI explanation
            if data:
                st.subheader("🤖 AI Explanation")
                with st.spinner("Generating AI explanation..."):
                    explanation = generate_explanation(file_format, geometry_type, data)
                st.markdown(explanation)

        except Exception as e:
            st.error(f"❌ Error analyzing the file: {str(e)}")

    # Clean up temporary file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# Add FAQ and Differences sections outside the upload block
st.markdown("---")
with st.expander("❓ Frequently Asked Questions (FAQ)"):
    st.markdown("""
    **Q: What is the difference between STEP and STL files?**  
    A: STEP files use B-Rep for exact geometry, ideal for engineering. STL files use mesh for approximations, great for 3D printing.

    **Q: Why can't I see a 3D preview for STEP files?**  
    A: STEP previews require advanced CAD libraries. For this POC, only STL previews are available.

    **Q: Is the AI explanation always accurate?**  
    A: It's generated by a language model based on your file data. For complex cases, consult a CAD expert.

    **Q: Can I upload large files?**  
    A: Up to 200MB is supported, but processing may take time for large meshes.

    **Q: How does this help in manufacturing?**  
    A: It helps choose the right format to avoid accuracy loss or inefficiencies in production.
    """)

with st.expander("🔄 B-Rep vs Mesh: Basic Differences"):
    st.markdown("""
    **B-Rep (Boundary Representation)**:  
    - **What it is**: Exact mathematical surfaces (planes, cylinders, etc.).  
    - **Pros**: High precision, editable, parametric.  
    - **Cons**: Larger files, slower processing, needs CAD software.  
    - **Best for**: Design, simulations, CNC machining.  

    **Mesh (Triangle Mesh)**:  
    - **What it is**: Approximates surfaces with flat triangles.  
    - **Pros**: Simple, fast, good for visualization.  
    - **Cons**: Approximate (faceting), hard to edit precisely.  
    - **Best for**: 3D printing, gaming, animations.  

    **Key Difference**: B-Rep is like a perfect blueprint; Mesh is like a pixelated image. Choose based on your needs for accuracy vs. speed!
    """)