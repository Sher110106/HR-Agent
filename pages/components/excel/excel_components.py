"""Excel-specific components for analysis and display."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from agents import SheetCatalogAgent, ColumnIndexerAgent

def render_sheet_catalog(sheet_catalog_agent: SheetCatalogAgent):
    """Render the sheet catalog information."""
    # Entire function body removed to eliminate summary and details sections
    pass

def render_column_index(column_indexer_agent: ColumnIndexerAgent):
    """Render the column index information."""
    st.subheader("🔍 Column Index")
    
    # Get column summary
    column_summary = column_indexer_agent.get_column_summary()
    
    if not column_summary:
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 All Columns", "🔗 Common Columns", "🔑 Join Keys"])
    
    with tab1:
        # All columns summary
        all_columns_data = []
        for col_name, info in column_summary.items():
            all_columns_data.append({
                "Column": col_name,
                "Sheets": len(info['sheets']),
                "Data Types": ", ".join(info['data_types']),
                "Avg Unique Ratio": f"{info['avg_unique_ratio']:.2f}",
                "Sample Values": ", ".join(str(v)[:20] for v in info['sample_values'][:3])
            })
        
        if all_columns_data:
            all_columns_df = pd.DataFrame(all_columns_data)
            st.dataframe(all_columns_df, width='stretch')
    
    with tab2:
        # Common columns (appearing in multiple sheets)
        common_columns = column_indexer_agent.find_common_columns(min_sheets=2)
        
        if common_columns:
            common_data = []
            for col_name, refs in common_columns.items():
                sheets = [ref.sheet_name for ref in refs]
                common_data.append({
                    "Column": col_name,
                    "Sheets": ", ".join(sheets),
                    "Count": len(sheets),
                    "Data Types": ", ".join(set(ref.data_type for ref in refs))
                })
            
            common_df = pd.DataFrame(common_data)
            st.dataframe(common_df, width='stretch')
        else:
            st.info("No columns found in multiple sheets.")
    
    with tab3:
        # Potential join keys
        potential_keys = column_indexer_agent.find_potential_join_keys()
        
        if potential_keys:
            keys_data = []
            for key in potential_keys:
                refs = column_indexer_agent.get_column_refs(key)
                keys_data.append({
                    "Join Key": key,
                    "Sheets": ", ".join(ref.sheet_name for ref in refs),
                    "Data Type": ", ".join(set(ref.data_type for ref in refs)),
                    "Avg Unique Ratio": f"{sum(ref.unique_count / max(1, ref.unique_count + ref.null_count) for ref in refs) / len(refs):.2f}"
                })
            
            keys_df = pd.DataFrame(keys_data)
            st.dataframe(keys_df, width='stretch')
        else:
            # Debug information
            with st.expander("🔍 Advanced: Column Analysis", expanded=False):
                st.markdown("**All columns in multiple sheets:**")
                common_columns = column_indexer_agent.find_common_columns(min_sheets=2)
                if common_columns:
                    for col_name, refs in common_columns.items():
                        st.markdown(f"- **{col_name}**: {len(refs)} sheets")
                        for ref in refs:
                            st.markdown(f"  - {ref.sheet_name}: {ref.data_type}, {ref.unique_count} unique, {ref.null_count} null")
                else:
                    st.markdown("No columns found in multiple sheets.")

def render_semantic_layer(column_indexer_agent: ColumnIndexerAgent):
    """Render the semantic layer management interface."""
    import logging
    logger = logging.getLogger(__name__)
    
    st.subheader("🏷️ Semantic Layer")
    
    # Get potential join keys for tagging
    potential_keys = column_indexer_agent.find_potential_join_keys()
    
    if not potential_keys:
        # Show common columns as fallback
        common_columns = column_indexer_agent.find_common_columns(min_sheets=2)
        if common_columns:
            st.markdown("**Available common columns:**")
            for col_name, refs in common_columns.items():
                sheets = [ref.sheet_name for ref in refs]
                st.markdown(f"- **{col_name}**: {', '.join(sheets)}")
            
            # Allow manual selection from common columns
            st.markdown("**Manual Join Key Selection**")
            manual_key = st.selectbox(
                "Select a join key from common columns:",
                options=["None"] + list(common_columns.keys()),
                help="Manually select a column that appears in multiple sheets"
            )
            
            if manual_key != "None":
                column_indexer_agent.add_semantic_tag("primary_join_key", manual_key)
                logger.info(f"Set primary join key: {manual_key}")
        return
    
    # Primary join key selection
    st.markdown("**Primary Join Key**")
    primary_key = st.selectbox(
        "Select the primary key for joining sheets:",
        options=["None"] + potential_keys,
        help="This key will be used as the default for joining sheets"
    )
    
    if primary_key != "None":
        column_indexer_agent.add_semantic_tag("primary_join_key", primary_key)
        logger.info(f"Set primary join key: {primary_key}")
        
        # Show details about the selected key
        refs = column_indexer_agent.get_column_refs(primary_key)
        if refs:
            st.markdown("**Join Key Details:**")
            for ref in refs:
                uniqueness_ratio = ref.unique_count / max(1, ref.unique_count + ref.null_count)
                st.markdown(f"- **{ref.sheet_name}**: {ref.data_type}, {ref.unique_count} unique values ({uniqueness_ratio:.1%} uniqueness)")
    
    # Show current semantic tags
    semantic_layer = column_indexer_agent.semantic_layer
    if semantic_layer:
        st.markdown("**Current Tags:**")
        for tag, value in semantic_layer.items():
            st.markdown(f"- **{tag}:** {value}")

def render_phase3_features(sheet_catalog_agent: SheetCatalogAgent, column_indexer_agent: ColumnIndexerAgent):
    """Render Phase 3: Resilience & Polish features."""
    st.subheader("🚀 Phase 3: Advanced Features")
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🛡️ Error Handling", "💾 Data Quality", "📤 Export"])
    
    with tab1:
        st.markdown("**Performance Monitoring**")
        
        # Get performance metrics
        if hasattr(sheet_catalog_agent, 'performance_monitor'):
            perf_summary = sheet_catalog_agent.performance_monitor.get_performance_summary()
            
            if perf_summary.get('status') != 'No queries recorded':
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Queries", perf_summary.get('total_queries', 0))
                with col2:
                    st.metric("Avg Query Time", f"{perf_summary.get('avg_query_time', 0):.2f}s")
                with col3:
                    st.metric("Cache Hit Rate", f"{perf_summary.get('cache_hit_rate', 0)*100:.1f}%")
                with col4:
                    st.metric("Memory Usage", f"{perf_summary.get('current_memory_usage_mb', 0):.1f} MB")
                
                # Performance actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🧹 Memory Cleanup", width='stretch'):
                        from utils.excel_performance import memory_cleanup
                        memory_usage = memory_cleanup()
                        logger.info(f"Memory cleanup completed. Current usage: {memory_usage:.1f} MB")
                        st.rerun()
                
                with col2:
                    if st.button("📊 Cache Stats", width='stretch'):
                        if hasattr(sheet_catalog_agent, 'cache'):
                            cache_stats = sheet_catalog_agent.cache.get_stats()
                            st.json(cache_stats)
        
        # Cache management
        if hasattr(sheet_catalog_agent, 'cache'):
            st.markdown("**Cache Management**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear Cache", width='stretch'):
                    sheet_catalog_agent.cache.clear()
                    logger.info("Cache cleared")
                    st.rerun()
            
            with col2:
                if st.button("📋 Cache Info", width='stretch'):
                    cache_stats = sheet_catalog_agent.cache.get_stats()
                    st.json(cache_stats)
    
    with tab2:
        st.markdown("**Error Handling & Recovery**")
        
        # Get error summary
        if hasattr(sheet_catalog_agent, 'error_handler'):
            error_summary = sheet_catalog_agent.error_handler.get_error_summary()
            
            if error_summary['total_errors'] > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Errors", error_summary['total_errors'])
                with col2:
                    st.metric("Critical Errors", error_summary['by_severity'].get('critical', 0))
                with col3:
                    st.metric("High Severity", error_summary['by_severity'].get('high', 0))
                
                # Show recent errors
                if error_summary['recent_errors']:
                    with st.expander("**Recent Errors**", expanded=False):
                        for error in error_summary['recent_errors'][-3:]:
                            st.text(f"• {error}")
        
        # Error handling actions
        st.markdown("**Error Recovery Actions:**")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retry Failed Operations", width='stretch'):
                logger.info("Retry functionality triggered")
        
        with col2:
            if st.button("📋 Error Log", width='stretch'):
                if hasattr(sheet_catalog_agent, 'error_handler'):
                    error_summary = sheet_catalog_agent.error_handler.get_error_summary()
                    st.json(error_summary)
    
    with tab3:
        st.markdown("**Data Quality Assessment**")
        
        if hasattr(column_indexer_agent, 'query_engine'):
            # Quality assessment for each sheet
            for sheet_name, df in sheet_catalog_agent.sheet_catalog.items():
                with st.expander(f"📋 {sheet_name} Quality Report", expanded=False):
                    try:
                        quality_report = column_indexer_agent.query_engine.validate_data_quality(df, sheet_name)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Quality Score", f"{quality_report['quality_score']:.1f}/100")
                        with col2:
                            st.metric("Total Rows", quality_report['total_rows'])
                        with col3:
                            st.metric("Null Rows", quality_report['duplicate_rows'])
                        with col4:
                            st.metric("Memory (MB)", f"{quality_report['memory_usage_mb']:.1f}")
                        
                        # Show quality issues
                        if quality_report['data_type_issues']:
                            with st.expander("**Data Type Issues**", expanded=False):
                                for issue in quality_report['data_type_issues']:
                                    st.markdown(f"- **{issue['column']}:** {', '.join(issue['issues'])}")
                        
                        if quality_report['outliers']:
                            with st.expander("**Outliers Detected**", expanded=False):
                                for col, outlier_info in quality_report['outliers'].items():
                                    st.markdown(f"- **{col}:** {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)")
                        
                    except Exception as e:
                        st.error(f"Error generating quality report: {e}")
    
    with tab4:
        st.markdown("**Export Capabilities**")
        
        # Export options
        export_format = st.selectbox("Export Format:", ["csv", "excel", "json", "docx"])
        
        if st.button("📤 Export Current Data", width='stretch'):
            if hasattr(column_indexer_agent, 'query_engine'):
                try:
                    # Export the first sheet as example
                    first_sheet = list(sheet_catalog_agent.sheet_catalog.values())[0]
                    
                    if export_format == "docx":
                        # Handle DOCX export separately
                        from utils.docx_utils import dataframe_to_docx_table
                        import pandas as pd
                        import io
                        
                        # Get the data as DataFrame first
                        file_content, filename = column_indexer_agent.query_engine.export_results(
                            first_sheet, format="csv"
                        )
                        
                        # Convert CSV content to DataFrame
                        csv_buffer = io.StringIO(file_content)
                        df = pd.read_csv(csv_buffer)
                        
                        # Convert to DOCX
                        docx_data = dataframe_to_docx_table(df, title=f"Excel Data Export - {first_sheet}")
                        docx_filename = filename.replace('.csv', '.docx')
                        
                        st.download_button(
                            label=f"📥 Download {docx_filename}",
                            data=docx_data,
                            file_name=docx_filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    else:
                        # Handle other formats normally
                        file_content, filename = column_indexer_agent.query_engine.export_results(
                            first_sheet, format=export_format
                        )
                        
                        st.download_button(
                            label=f"📥 Download {filename}",
                            data=file_content,
                            file_name=filename,
                            mime="application/octet-stream"
                        )
                    
                except Exception as e:
                    st.error(f"Export failed: {e}")
        
        # Advanced export options
        st.markdown("**Advanced Export Options:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export All Sheets", width='stretch'):
                logger.info("Multi-sheet export triggered")
        
        with col2:
            if st.button("📈 Export Analysis Results", width='stretch'):
                logger.info("Analysis results export triggered")
