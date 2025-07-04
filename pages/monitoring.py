import streamlit as st
from utils.health_monitor import (
    get_health_status, perform_health_check, get_detailed_health_status
)
from utils.metrics import get_metrics_summary
from utils.circuit_breaker import get_all_circuit_breaker_stats
from utils.cache import get_cache_stats, cleanup_all_caches

def export_health_report(filename: str):
    """Export health report to a JSON file."""
    import json
    
    health_data = get_detailed_health_status()
    with open(filename, 'w') as f:
        json.dump(health_data, f, indent=2, default=str)

def monitoring_dashboard():
    """System monitoring and health dashboard."""
    st.title("🔧 System Monitoring Dashboard")
    
    # Health Status
    st.header("🏥 System Health")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Health Check"):
            perform_health_check()
    
    with col2:
        if st.button("📊 Export Health Report"):
            export_health_report("health_report.json")
            st.success("Health report exported to health_report.json")
    
    with col3:
        if st.button("🗑️ Clear Cache"):
            cleanup_all_caches()
            st.success("Cache cleanup completed")
    
    # Get health status
    health_status = get_health_status()
    
    # Overall status
    status_color = {
        "healthy": "🟢",
        "degraded": "🟡", 
        "unhealthy": "🔴",
        "unknown": "⚪"
    }
    
    overall_status = health_status.get("overall_status", "unknown")
    st.metric(
        "Overall System Status",
        f"{status_color.get(overall_status, '⚪')} {overall_status.upper()}",
        f"Uptime: {health_status.get('uptime_minutes', 0):.1f} min"
    )
    
    # Health checks
    if "checks" in health_status:
        st.subheader("📋 Health Checks")
        for check_name, check_data in health_status["checks"].items():
            with st.expander(f"{status_color.get(check_data['status'], '⚪')} {check_name.replace('_', ' ').title()}"):
                st.write(f"**Status:** {check_data['status']}")
                st.write(f"**Message:** {check_data['message']}")
                if check_data.get('response_time_ms'):
                    st.write(f"**Response Time:** {check_data['response_time_ms']:.1f}ms")
                if check_data.get('metadata'):
                    st.json(check_data['metadata'])
    
    # Performance Metrics
    st.header("📈 Performance Metrics")
    
    # Get metrics summary
    metrics = get_metrics_summary(hours_back=1)
    
    if metrics.get("total_events", 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "API Calls",
                metrics["api_calls"]["count"],
                f"{metrics['api_calls']['success_rate']:.1%} success rate"
            )
        
        with col2:
            st.metric(
                "Avg Response Time",
                f"{metrics['api_calls']['avg_response_time_ms']:.0f}ms",
                f"{metrics['api_calls']['total_tokens']} tokens"
            )
        
        with col3:
            st.metric(
                "Code Executions", 
                metrics["code_executions"]["count"],
                f"{metrics['code_executions']['success_rate']:.1%} success rate"
            )
        
        with col4:
            st.metric(
                "Error Rate",
                f"{metrics['errors']['error_rate']:.1%}",
                f"{metrics['errors']['count']} errors"
            )
        
        # Error breakdown
        if metrics["errors"]["count"] > 0:
            st.subheader("❌ Error Breakdown")
            for error in metrics["errors"]["top_error_types"]:
                st.write(f"• **{error['type']}**: {error['count']} occurrences")
    
    else:
        st.info("No performance metrics available yet. Start using the application to see metrics.")
    
    # Circuit Breaker Status
    st.header("🔧 Circuit Breaker Status")
    
    breaker_stats = get_all_circuit_breaker_stats()
    if breaker_stats:
        for name, stats in breaker_stats.items():
            state_color = {
                "closed": "🟢",
                "half_open": "🟡",
                "open": "🔴"
            }
            
            with st.expander(f"{state_color.get(stats['state'], '⚪')} {name.replace('_', ' ').title()}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**State:** {stats['state']}")
                    st.write(f"**Failure Count:** {stats['failure_count']}")
                    st.write(f"**Success Count:** {stats['success_count']}")
                
                with col2:
                    st.write("**Configuration:**")
                    st.json(stats['config'])
    
    # Cache Statistics
    st.header("💾 Cache Statistics")
    
    cache_stats = get_cache_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Code Cache")
        code_stats = cache_stats["code_cache"]
        st.metric("Cache Size", f"{code_stats['size']}/{code_stats['max_size']}")
        st.metric("Hit Rate", f"{code_stats['hit_rate']:.1%}")
        st.metric("Total Requests", code_stats['total_requests'])
        
        if code_stats.get('memory_usage_estimate'):
            st.metric("Memory Usage", f"{code_stats['memory_usage_estimate']['total_mb']:.1f} MB")
    
    with col2:
        st.subheader("Analysis Cache") 
        analysis_stats = cache_stats["analysis_cache"]
        st.metric("Cache Size", f"{analysis_stats['size']}/{analysis_stats['max_size']}")
        st.metric("Hit Rate", f"{analysis_stats['hit_rate']:.1%}")
        st.metric("Total Requests", analysis_stats['total_requests'])
        
        if analysis_stats.get('memory_usage_estimate'):
            st.metric("Memory Usage", f"{analysis_stats['memory_usage_estimate']['total_mb']:.1f} MB")
    
    # Popular Patterns
    if cache_stats.get("popular_patterns"):
        st.subheader("🔥 Popular Query Patterns")
        for pattern, count in cache_stats["popular_patterns"]:
            st.write(f"• **{pattern}**: {count} uses") 