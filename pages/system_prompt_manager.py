"""
System Prompt Manager page for the Data Analysis Agent.
Provides UI components for creating, editing, and managing system prompts.
"""

import streamlit as st
from typing import List, Dict
from utils.system_prompts import get_prompt_manager, SystemPrompt, reinitialize_prompt_manager
from agents.memory import SystemPromptMemoryAgent
from utils.logging_config import get_logger

logger = get_logger(__name__)


def render_quick_demo_section():
    """Render a quick demo section for testing system prompts."""
    st.subheader("🚀 Quick Demo")
    st.markdown("Try this demo prompt to see system prompts in action!")
    
    prompt_manager = get_prompt_manager()
    
    demo_prompt = """You are a data analysis assistant with a friendly, encouraging personality.

Your approach:
- Always start responses with an enthusiastic greeting
- Use simple, non-technical language that anyone can understand
- Include relevant emojis to make responses more engaging
- End with a motivational statement about data insights
- Focus on practical, everyday business applications

Communication style: Friendly, accessible, and motivational."""

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎭 Create Demo Prompt", use_container_width=True):
            success = prompt_manager.create_prompt(
                name="Friendly Data Assistant",
                prompt=demo_prompt,
                description="A friendly, approachable assistant that makes data analysis accessible to everyone",
                category="custom",
                tags=["demo", "friendly", "beginner-friendly"]
            )
            if success:
                st.success("✅ Demo prompt created! Try it in the Data Analysis page.")
            else:
                st.warning("⚠️ Demo prompt already exists!")
    
    with col2:
        if st.button("🎯 Activate Demo Prompt", use_container_width=True):
            system_prompt_agent = SystemPromptMemoryAgent()
            success = system_prompt_agent.set_active_prompt("Friendly Data Assistant")
            if success:
                st.success("✅ Demo prompt activated! Go try it in Data Analysis.")
            else:
                st.error("❌ Please create the demo prompt first.")
    
    with st.expander("👀 Preview Demo Prompt", expanded=False):
        st.code(demo_prompt, language="text")
        st.markdown("**What this does:** This prompt makes the AI respond in a friendly, encouraging way with emojis and simple language - perfect for testing how system prompts change AI behavior!")


def render_prompt_selector():
    """Render the system prompt selector component."""
    prompt_manager = get_prompt_manager()
    system_prompt_agent = SystemPromptMemoryAgent()
    
    st.subheader("🎯 Active System Prompt")
    
    # Get current active prompt
    active_prompt = prompt_manager.get_active_prompt()
    current_selection = active_prompt.name if active_prompt else "None"
    
    # Get all available prompts
    all_prompts = prompt_manager.list_prompts()
    prompt_options = ["None"] + [p.name for p in all_prompts]
    
    # Create selectbox
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_prompt = st.selectbox(
            "Choose a system prompt to apply to all AI interactions:",
            options=prompt_options,
            index=prompt_options.index(current_selection) if current_selection in prompt_options else 0,
            key="active_prompt_selector"
        )
    
    with col2:
        if st.button("🔄 Apply", key="apply_prompt"):
            if selected_prompt == "None":
                system_prompt_agent.clear_active_prompt()
                st.success("✅ System prompt cleared - now using default AI behavior")
                st.balloons()  # Celebration effect
                st.rerun()
            else:
                with st.spinner(f"Activating {selected_prompt}..."):
                    success = system_prompt_agent.set_active_prompt(selected_prompt)
                    if success:
                        st.success(f"🎯 Successfully activated: **{selected_prompt}**")
                        st.balloons()  # Celebration effect
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to activate: {selected_prompt}")
                        st.error("💡 **Troubleshooting:** Try refreshing the page or check if the prompt exists")
    
    # Show active prompt info
    if active_prompt:
        with st.expander("📋 Active Prompt Details", expanded=False):
            st.markdown(f"**Category:** {active_prompt.category}")
            st.markdown(f"**Description:** {active_prompt.description}")
            st.markdown(f"**Usage Count:** {active_prompt.usage_count}")
            if active_prompt.tags:
                st.markdown(f"**Tags:** {', '.join(active_prompt.tags)}")
            
            st.markdown("**Prompt Text:**")
            st.code(active_prompt.prompt, language="text")


def render_prompt_templates():
    """Render the prompt templates section."""
    prompt_manager = get_prompt_manager()
    
    st.subheader("📚 Prompt Templates")
    
    # Get templates by category
    all_prompts = prompt_manager.list_prompts()
    categories = prompt_manager.get_categories()
    
    # Create tabs for each category
    if categories:
        tabs = st.tabs([f"📁 {cat.replace('_', ' ').title()}" for cat in categories])
        
        for i, category in enumerate(categories):
            with tabs[i]:
                category_prompts = [p for p in all_prompts if p.category == category]
                
                if not category_prompts:
                    st.info(f"No prompts in {category} category")
                    continue
                
                for prompt in category_prompts:
                    with st.expander(f"🎯 {prompt.name}", expanded=False):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**Description:** {prompt.description}")
                            if prompt.tags:
                                st.markdown(f"**Tags:** {', '.join(prompt.tags)}")
                            st.markdown(f"**Usage:** {prompt.usage_count} times")
                        
                        with col2:
                            if st.button("🎯 Activate", key=f"activate_{prompt.name}"):
                                system_prompt_agent = SystemPromptMemoryAgent()
                                success = system_prompt_agent.set_active_prompt(prompt.name)
                                if success:
                                    st.success("✅ Activated!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to activate")
                        
                        with col3:
                            if prompt.category == "custom":
                                if st.button("🗑️ Delete", key=f"delete_{prompt.name}"):
                                    if prompt_manager.delete_prompt(prompt.name):
                                        st.success("✅ Deleted!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Cannot delete")
                        
                        st.markdown("**Prompt Text:**")
                        st.code(prompt.prompt, language="text")


def render_create_prompt_form():
    """Render the create new prompt form."""
    prompt_manager = get_prompt_manager()
    
    st.subheader("➕ Create Custom Prompt")
    
    with st.form("create_prompt_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(
                "Prompt Name*",
                placeholder="e.g., Marketing Analyst",
                help="Unique name for your custom prompt"
            )
            
            category = st.selectbox(
                "Category",
                options=["custom", "analysis", "creative", "strategy", "technical", "human_resources"],
                index=0,
                help="Category to organize your prompt"
            )
        
        with col2:
            description = st.text_area(
                "Description*",
                placeholder="Brief description of what this prompt does...",
                height=100,
                help="Describe the purpose and use case"
            )
            
            tags_input = st.text_input(
                "Tags (optional)",
                placeholder="analysis, marketing, strategy",
                help="Comma-separated tags for easy searching"
            )
        
        prompt_text = st.text_area(
            "System Prompt*",
            placeholder="""You are an expert marketing analyst with deep knowledge in customer behavior and market research.

Your approach:
- Analyze customer data with a focus on actionable insights
- Consider market trends and competitive landscape
- Provide clear recommendations for marketing strategies
- Use data-driven reasoning for all conclusions

Communication style: Professional, strategic, and results-oriented.""",
            height=200,
            help="The actual system prompt text that will be sent to the AI"
        )
        
        submitted = st.form_submit_button("🚀 Create Prompt")
        
        if submitted:
            if not name or not description or not prompt_text:
                st.error("❌ Please fill in all required fields (marked with *)")
            elif name in [p.name for p in prompt_manager.list_prompts()]:
                st.error(f"❌ Prompt named '{name}' already exists")
            else:
                tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()] if tags_input else []
                
                success = prompt_manager.create_prompt(
                    name=name,
                    prompt=prompt_text,
                    description=description,
                    category=category,
                    tags=tags
                )
                
                if success:
                    st.success(f"✅ Created prompt: {name}")
                    st.rerun()
                else:
                    st.error("❌ Failed to create prompt")


def render_import_export_section():
    """Render import/export functionality."""
    prompt_manager = get_prompt_manager()
    
    st.subheader("📁 Import/Export Prompts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📤 Export Prompts**")
        
        if st.button("💾 Export All Prompts"):
            filename = "exported_prompts.json"
            success = prompt_manager.export_prompts(filename)
            if success:
                st.success(f"✅ Exported to {filename}")
                
                # Provide download buttons
                try:
                    with open(filename, "r") as f:
                        json_data = f.read()
                    
                    # Create two columns for download options
                    json_col, docx_col = st.columns(2)
                    
                    with json_col:
                        st.download_button(
                            "⬇️ JSON",
                            data=json_data,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with docx_col:
                        # Convert JSON to readable DOCX format
                        from utils.docx_utils import text_to_docx
                        import json
                        
                        # Parse JSON and create readable text
                        prompts_data = json.loads(json_data)
                        readable_text = "Exported System Prompts\n\n"
                        
                        for prompt_name, prompt_info in prompts_data.items():
                            readable_text += f"## {prompt_name}\n"
                            if 'description' in prompt_info:
                                readable_text += f"Description: {prompt_info['description']}\n\n"
                            if 'prompt' in prompt_info:
                                readable_text += f"Prompt:\n{prompt_info['prompt']}\n\n"
                            if 'tags' in prompt_info:
                                readable_text += f"Tags: {', '.join(prompt_info['tags'])}\n\n"
                            readable_text += "---\n\n"
                        
                        docx_data = text_to_docx(readable_text, title="Exported System Prompts")
                        st.download_button(
                            "📝 DOCX",
                            data=docx_data,
                            file_name="exported_prompts.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                        
                except Exception as e:
                    st.error(f"❌ Export file error: {e}")
            else:
                st.error("❌ Export failed")
    
    with col2:
        st.markdown("**📥 Import Prompts**")
        
        uploaded_file = st.file_uploader(
            "Choose JSON file",
            type="json",
            help="Upload a JSON file containing prompt definitions"
        )
        
        overwrite = st.checkbox(
            "Overwrite existing prompts",
            help="Replace existing prompts with same names"
        )
        
        if uploaded_file and st.button("📥 Import Prompts"):
            try:
                # Save uploaded file temporarily
                temp_filename = "temp_import.json"
                with open(temp_filename, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                imported_count = prompt_manager.import_prompts(temp_filename, overwrite)
                
                if imported_count > 0:
                    st.success(f"✅ Imported {imported_count} prompts")
                    st.rerun()
                else:
                    st.warning("⚠️ No new prompts imported")
                
                # Cleanup
                import os
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
            except Exception as e:
                st.error(f"❌ Import failed: {e}")


def render_prompt_search():
    """Render prompt search functionality."""
    prompt_manager = get_prompt_manager()
    
    st.subheader("🔍 Search Prompts")
    
    search_query = st.text_input(
        "Search prompts by name, description, or tags:",
        placeholder="Enter search terms...",
        key="prompt_search"
    )
    
    if search_query:
        results = prompt_manager.search_prompts(search_query)
        
        if results:
            st.markdown(f"**Found {len(results)} matching prompts:**")
            
            for prompt in results:
                with st.expander(f"🎯 {prompt.name} ({prompt.category})", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {prompt.description}")
                        if prompt.tags:
                            st.markdown(f"**Tags:** {', '.join(prompt.tags)}")
                        st.markdown(f"**Usage:** {prompt.usage_count} times")
                    
                    with col2:
                        if st.button("🎯 Activate", key=f"search_activate_{prompt.name}"):
                            system_prompt_agent = SystemPromptMemoryAgent()
                            success = system_prompt_agent.set_active_prompt(prompt.name)
                            if success:
                                st.success("✅ Activated!")
                                st.rerun()
                    
                    st.code(prompt.prompt, language="text")
        else:
            st.info("No prompts found matching your search.")


def system_prompt_manager_page():
    """Main system prompt manager page."""
    st.title("🎯 System Prompt Manager")
    st.markdown("Create and manage custom system prompts to guide AI behavior")
    
    # Force reinitialize to ensure all defaults are loaded
    reinitialize_prompt_manager()
    
    # Enhanced status indicator with better visuals
    prompt_manager = get_prompt_manager()
    active_prompt = prompt_manager.get_active_prompt()
    
    if active_prompt:
        # Create a prominent status card for active prompt
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin: 0; color: white;">🎯 ACTIVE SYSTEM PROMPT</h3>
            <h2 style="margin: 10px 0 5px 0; color: white;">{}</h2>
            <p style="margin: 0; opacity: 0.9;">Category: {} | Used {} times</p>
        </div>
        """.format(active_prompt.name, active_prompt.category.title(), active_prompt.usage_count), 
        unsafe_allow_html=True)
        
        # Quick action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Status", key="refresh_status"):
                st.rerun()
        with col2:
            if st.button("❌ Clear Active", key="clear_active"):
                prompt_manager.clear_active_prompt()
                st.success("✅ Cleared active prompt")
                st.rerun()
        with col3:
            st.markdown(f"**Description:** {active_prompt.description}")
    else:
        # Prominent indicator for no active prompt
        st.markdown("""
        <div style="
            background: linear-gradient(90deg, #FF9800, #F57C00);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h3 style="margin: 0; color: white;">⚠️ NO ACTIVE SYSTEM PROMPT</h3>
            <h4 style="margin: 10px 0 5px 0; color: white;">Using Default AI Behavior</h4>
            <p style="margin: 0; opacity: 0.9;">Select a prompt below to customize AI responses</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Refresh Status", key="refresh_status_none"):
            st.rerun()
    
    st.markdown("---")
    
    # Quick demo section at the top
    render_quick_demo_section()
    
    st.markdown("---")
    
    # Main sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Select Prompt", 
        "📚 Browse Templates", 
        "➕ Create New", 
        "🔍 Search", 
        "📁 Import/Export"
    ])
    
    with tab1:
        render_prompt_selector()
    
    with tab2:
        render_prompt_templates()
    
    with tab3:
        render_create_prompt_form()
    
    with tab4:
        render_prompt_search()
    
    with tab5:
        render_import_export_section()


# For backwards compatibility and direct page access
def main():
    """Entry point when running as standalone page."""
    system_prompt_manager_page()


if __name__ == "__main__":
    main() 