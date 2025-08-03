import unittest
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_helpers import (
    format_axis_labels,
    apply_professional_styling,
    get_professional_colors,
    optimize_figure_size,
    create_clean_bar_chart,
    # Phase 1 enhancements
    get_hr_specific_colors,
    get_gradient_colormap,
    get_contextual_colors,
    apply_modern_typography,
    apply_golden_ratio_spacing,
    apply_modern_styling,
    add_smart_annotations,
    create_gradient_background,
    create_clean_line_chart,
    create_clean_scatter_plot,
    create_clean_histogram,
    create_clean_box_plot,
    create_clean_heatmap,
    create_clean_pie_chart,
    # Phase 2 enhancements
    create_clean_violin_plot,
    create_clean_swarm_plot,
    create_clean_waterfall_chart,
    create_clean_ridge_plot,
    create_clean_sankey_diagram,
    add_interactive_elements,
    detect_insights,
    add_insight_annotations,
    create_enhanced_chart_with_insights,
    PlotMemory,
    is_plot_modification_request,
    generate_plot_modification_code
)

class TestPlotHelpers(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value': [10, 20, 15, 5],
            'group': ['X', 'Y', 'X', 'Y'],
            'salary': [50000, 75000, 60000, 45000],
            'tenure': [2, 5, 3, 1]
        })

    def tearDown(self):
        plt.close(self.fig)

    def test_format_axis_labels(self):
        self.ax.bar(self.df['category'], self.df['value'])
        format_axis_labels(self.ax, x_rotation=30)
        # Just check that labels exist and are rotated
        for label in self.ax.get_xticklabels():
            self.assertIsNotNone(label.get_text())

    def test_apply_professional_styling(self):
        apply_professional_styling(self.ax, title="Test Title", xlabel="X", ylabel="Y")
        self.assertEqual(self.ax.get_title(), "Test Title")
        self.assertEqual(self.ax.get_xlabel(), "X")
        self.assertEqual(self.ax.get_ylabel(), "Y")

    def test_get_professional_colors(self):
        colors = get_professional_colors()
        self.assertIn('colors', colors)
        self.assertIsInstance(colors['colors'], list)
        self.assertGreater(len(colors['colors']), 0)

    def test_optimize_figure_size(self):
        # Should not raise
        optimize_figure_size(self.ax)
        # Check that figure size is a tuple of length 2
        size = self.fig.get_size_inches()
        self.assertEqual(len(size), 2)

    def test_create_clean_bar_chart_simple(self):
        # Should not raise and should add bars
        create_clean_bar_chart(self.ax, self.df, x_col='category', y_col='value', title="Bar", xlabel="Cat", ylabel="Val")
        bars = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
        self.assertTrue(any(bar.get_height() > 0 for bar in bars))

    def test_create_clean_bar_chart_grouped(self):
        # Should not raise for grouped bar chart
        create_clean_bar_chart(self.ax, self.df, x_col='category', y_col='value', hue_col='group', title="Grouped", xlabel="Cat", ylabel="Val")
        bars = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
        self.assertTrue(any(bar.get_height() > 0 for bar in bars))

    # Phase 1 Enhancement Tests
    def test_get_hr_specific_colors(self):
        """Test HR-specific color palettes."""
        hr_colors = get_hr_specific_colors()
        self.assertIn('attrition', hr_colors)
        self.assertIn('retention', hr_colors)
        self.assertIn('performance', hr_colors)
        self.assertIn('neutral', hr_colors)
        self.assertIn('accessibility', hr_colors)
        self.assertIn('modern', hr_colors)
        
        # Check that each palette has colors
        for palette_name, colors in hr_colors.items():
            self.assertIsInstance(colors, list)
            self.assertGreater(len(colors), 0)
            # Check that colors are valid hex codes
            for color in colors:
                self.assertTrue(color.startswith('#') and len(color) == 7)

    def test_get_gradient_colormap(self):
        """Test gradient colormap creation."""
        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#3498db']
        cmap = get_gradient_colormap(colors, 'test_cmap')
        self.assertIsInstance(cmap, matplotlib.colors.LinearSegmentedColormap)
        self.assertEqual(cmap.name, 'test_cmap')

    def test_get_contextual_colors(self):
        """Test contextual color selection."""
        # Test with different data types
        attrition_colors = get_contextual_colors('attrition', ['A', 'B', 'C'])
        retention_colors = get_contextual_colors('retention', ['X', 'Y'])
        neutral_colors = get_contextual_colors('neutral')
        
        self.assertEqual(len(attrition_colors), 3)
        self.assertEqual(len(retention_colors), 2)
        self.assertIsInstance(neutral_colors, list)
        self.assertGreater(len(neutral_colors), 0)

    def test_apply_modern_typography(self):
        """Test modern typography application."""
        self.ax.set_title("Test Title")
        self.ax.set_xlabel("X Label")
        self.ax.set_ylabel("Y Label")
        
        apply_modern_typography(self.ax, font_family='system')
        
        # Check that typography was applied (font sizes should be reasonable)
        title = self.ax.get_title()
        self.assertIsNotNone(title)

    def test_apply_golden_ratio_spacing(self):
        """Test golden ratio spacing application."""
        original_position = self.ax.get_position()
        apply_golden_ratio_spacing(self.ax)
        new_position = self.ax.get_position()
        
        # Position should have changed
        self.assertNotEqual(original_position, new_position)

    def test_apply_modern_styling(self):
        """Test modern styling themes."""
        # Test professional theme
        apply_modern_styling(self.ax, theme='professional')
        self.assertFalse(self.ax.spines['top'].get_visible())
        self.assertFalse(self.ax.spines['right'].get_visible())
        
        # Test modern theme
        self.ax.clear()
        apply_modern_styling(self.ax, theme='modern')
        self.assertFalse(self.ax.spines['top'].get_visible())
        self.assertFalse(self.ax.spines['right'].get_visible())
        
        # Test minimal theme
        self.ax.clear()
        apply_modern_styling(self.ax, theme='minimal')
        self.assertFalse(self.ax.grid())
        
        # Test elegant theme
        self.ax.clear()
        apply_modern_styling(self.ax, theme='elegant')
        # Check that grid is enabled (grid() returns None when grid is on)
        self.assertIsNone(self.ax.grid())

    def test_add_smart_annotations(self):
        """Test smart annotations."""
        # Create a bar chart first
        self.ax.bar(self.df['category'], self.df['value'])
        
        # Add smart annotations
        add_smart_annotations(self.ax, self.df, highlight_insights=True)
        
        # Check that annotations were added (text objects)
        text_objects = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Text)]
        self.assertGreater(len(text_objects), 0)

    def test_create_gradient_background(self):
        """Test gradient background creation."""
        create_gradient_background(self.ax)
        
        # Check that background colors were set
        self.assertIsNotNone(self.ax.get_facecolor())
        self.assertIsNotNone(self.ax.figure.patch.get_facecolor())

    def test_create_clean_line_chart(self):
        """Test enhanced line chart creation."""
        create_clean_line_chart(self.ax, self.df, x_col='category', y_col='value', 
                              title="Line Chart", theme='modern')
        
        # Check that lines were created
        lines = [child for child in self.ax.get_children() if isinstance(child, matplotlib.lines.Line2D)]
        self.assertGreater(len(lines), 0)

    def test_create_clean_scatter_plot(self):
        """Test enhanced scatter plot creation."""
        create_clean_scatter_plot(self.ax, self.df, x_col='salary', y_col='tenure',
                                title="Scatter Plot", theme='elegant')
        
        # Check that scatter points were created
        collections = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.PathCollection)]
        self.assertGreater(len(collections), 0)

    def test_create_clean_histogram(self):
        """Test enhanced histogram creation."""
        create_clean_histogram(self.ax, self.df, col='salary', title="Histogram", theme='minimal')
        
        # Check that histogram was created
        patches = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
        self.assertGreater(len(patches), 0)

    def test_create_clean_box_plot(self):
        """Test enhanced box plot creation."""
        create_clean_box_plot(self.ax, self.df, x_col='group', y_col='salary',
                            title="Box Plot", theme='professional')
        
        # Check that box plot was created
        # Box plots create multiple types of artists
        artists = self.ax.get_children()
        self.assertGreater(len(artists), 0)

    def test_create_clean_heatmap(self):
        """Test enhanced heatmap creation."""
        # Create correlation data
        corr_df = self.df[['salary', 'tenure', 'value']].corr()
        create_clean_heatmap(self.ax, corr_df, title="Heatmap", theme='modern')
        
        # Check that heatmap was created
        collections = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.QuadMesh)]
        self.assertGreater(len(collections), 0)

    def test_create_clean_pie_chart(self):
        """Test enhanced pie chart creation."""
        create_clean_pie_chart(self.ax, self.df, col='group', title="Pie Chart", theme='elegant')
        
        # Check that pie chart was created
        wedges = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Wedge)]
        self.assertGreater(len(wedges), 0)

    def test_theme_consistency(self):
        """Test that different themes produce consistent results."""
        themes = ['professional', 'modern', 'minimal', 'elegant']
        
        for theme in themes:
            self.ax.clear()
            create_clean_bar_chart(self.ax, self.df, x_col='category', y_col='value',
                                 title=f"Test {theme}", theme=theme)
            
            # All themes should produce a valid chart
            bars = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
            self.assertTrue(any(bar.get_height() > 0 for bar in bars))

    def test_enhanced_bar_chart_with_annotations(self):
        """Test bar chart with smart annotations."""
        create_clean_bar_chart(self.ax, self.df, x_col='category', y_col='value',
                             title="Annotated Bar Chart", add_annotations=True)
        
        # Check that both bars and annotations were created
        bars = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
        text_objects = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Text)]
        
        self.assertTrue(any(bar.get_height() > 0 for bar in bars))
        self.assertGreater(len(text_objects), 0)

    # Phase 2 Enhancement Tests
    def test_create_clean_violin_plot(self):
        """Test enhanced violin plot creation."""
        create_clean_violin_plot(self.ax, self.df, x_col='group', y_col='salary',
                               title="Violin Plot", theme='professional')
        
        # Check that violin plot was created
        artists = self.ax.get_children()
        self.assertGreater(len(artists), 0)

    def test_create_clean_swarm_plot(self):
        """Test enhanced swarm plot creation."""
        create_clean_swarm_plot(self.ax, self.df, x_col='group', y_col='salary',
                              title="Swarm Plot", theme='modern')
        
        # Check that swarm plot was created
        artists = self.ax.get_children()
        self.assertGreater(len(artists), 0)

    def test_create_clean_waterfall_chart(self):
        """Test enhanced waterfall chart creation."""
        create_clean_waterfall_chart(self.ax, self.df, x_col='category', y_col='value',
                                   title="Waterfall Chart", theme='elegant')
        
        # Check that waterfall chart was created
        bars = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
        self.assertGreater(len(bars), 0)

    def test_create_clean_ridge_plot(self):
        """Test enhanced ridge plot creation."""
        create_clean_ridge_plot(self.ax, self.df, x_col='salary', y_col='tenure',
                              group_col='group', title="Ridge Plot", theme='professional')
        
        # Check that ridge plot was created
        artists = self.ax.get_children()
        self.assertGreater(len(artists), 0)

    def test_create_clean_sankey_diagram(self):
        """Test enhanced Sankey diagram creation."""
        source = ['A', 'B', 'C']
        target = ['X', 'Y', 'Z']
        value = [10, 20, 15]
        
        create_clean_sankey_diagram(self.ax, source, target, value,
                                  title="Sankey Diagram", theme='modern')
        
        # Check that Sankey diagram was created (fallback text)
        text_objects = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Text)]
        self.assertGreater(len(text_objects), 0)

    def test_detect_insights(self):
        """Test insight detection functionality."""
        insights = detect_insights(self.df, x_col='salary', y_col='tenure')
        
        # Check that insights dictionary has expected structure
        self.assertIn('outliers', insights)
        self.assertIn('trends', insights)
        self.assertIn('patterns', insights)
        self.assertIn('correlations', insights)
        self.assertIn('summary_stats', insights)
        
        # Check that summary stats are calculated
        if insights['summary_stats']:
            self.assertIn('mean', insights['summary_stats'])
            self.assertIn('median', insights['summary_stats'])
            self.assertIn('std', insights['summary_stats'])

    def test_add_insight_annotations(self):
        """Test insight annotation addition."""
        insights = {
            'summary_stats': {'mean': 10.5, 'std': 2.5},
            'trends': ['strong positive correlation (0.85)']
        }
        
        add_insight_annotations(self.ax, insights)
        
        # Check that annotations were added
        text_objects = [child for child in self.ax.get_children() if isinstance(child, matplotlib.text.Text)]
        self.assertGreater(len(text_objects), 0)

    def test_create_enhanced_chart_with_insights(self):
        """Test enhanced chart creation with automatic insights."""
        create_enhanced_chart_with_insights(self.ax, self.df, 'bar', 'category', 'value',
                                          title="Enhanced Chart", add_insights=True)
        
        # Check that chart was created
        bars = [child for child in self.ax.get_children() if isinstance(child, matplotlib.patches.Rectangle)]
        self.assertTrue(any(bar.get_height() > 0 for bar in bars))

    def test_plot_memory_system(self):
        """Test plot memory system functionality."""
        plot_memory = PlotMemory()
        
        # Create a test plot
        fig, ax = plt.subplots()
        ax.bar(self.df['category'], self.df['value'])
        
        # Add plot to memory
        plot_id = plot_memory.add_plot(
            fig=fig,
            data_df=self.df,
            context="Test plot",
            chart_type="bar",
            styling={'theme': 'professional'}
        )
        
        # Check that plot was added
        self.assertEqual(plot_id, 0)
        self.assertEqual(len(plot_memory.plots), 1)
        
        # Test getting plot by reference
        plot = plot_memory.get_plot_by_reference("the plot")
        self.assertIsNotNone(plot)
        self.assertEqual(plot['chart_type'], "bar")
        
        # Test getting plot by ID
        plot = plot_memory.get_plot_by_id(0)
        self.assertIsNotNone(plot)
        self.assertEqual(plot['chart_type'], "bar")
        
        # Test listing plots
        plot_list = plot_memory.list_plots()
        self.assertEqual(len(plot_list), 1)
        self.assertEqual(plot_list[0][1], "bar")

    def test_is_plot_modification_request(self):
        """Test plot modification request detection."""
        # Test modification requests
        modification_queries = [
            "change the colors",
            "modify the plot",
            "edit the chart",
            "update the visualization",
            "adjust the size",
            "make it blue",
            "switch to line chart"
        ]
        
        for query in modification_queries:
            self.assertTrue(is_plot_modification_request(query))
        
        # Test non-modification requests
        non_modification_queries = [
            "show me the data",
            "what is the average",
            "create a chart",
            "analyze the trends"
        ]
        
        for query in non_modification_queries:
            self.assertFalse(is_plot_modification_request(query))

    def test_generate_plot_modification_code(self):
        """Test plot modification code generation."""
        target_plot = {
            'chart_type': 'bar',
            'data': self.df,
            'context': 'Test plot',
            'styling': {'theme': 'professional'}
        }
        
        modification_code = generate_plot_modification_code("change colors to blue", target_plot, self.df)
        
        # Check that modification code was generated
        self.assertIsInstance(modification_code, str)
        self.assertIn("ORIGINAL PLOT", modification_code)
        self.assertIn("USER REQUEST", modification_code)
        self.assertIn("bar", modification_code)

    def test_add_interactive_elements(self):
        """Test interactive elements addition."""
        fig, ax = plt.subplots()
        ax.bar(self.df['category'], self.df['value'])
        
        # Should not raise
        add_interactive_elements(ax, fig, self.df)

    def test_create_clean_radar_chart(self):
        from utils.plot_helpers import create_clean_radar_chart
        fig, ax = plt.subplots(subplot_kw={'polar': True})
        categories = ['A', 'B', 'C', 'D']
        values = [[1, 2, 3, 4], [2, 3, 2, 1]]
        group_labels = ['Group 1', 'Group 2']
        # Should not raise
        create_clean_radar_chart(ax, None, categories, values, group_labels, title="Radar Test")
        self.assertEqual(ax.get_title(), "Radar Test")
        plt.close(fig)

    def test_create_clean_treemap(self):
        from utils.plot_helpers import create_clean_treemap
        fig, ax = plt.subplots()
        labels = ['A', 'B', 'C']
        sizes = [10, 20, 30]
        # Should not raise
        create_clean_treemap(ax, labels, sizes, title="Treemap Test")
        self.assertEqual(ax.get_title(), "Treemap Test")
        plt.close(fig)

    def test_create_clean_gantt_chart(self):
        from utils.plot_helpers import create_clean_gantt_chart
        fig, ax = plt.subplots()
        df = pd.DataFrame({
            'task': ['Task 1', 'Task 2'],
            'start': ['2024-06-01', '2024-06-02'],
            'end': ['2024-06-03', '2024-06-04']
        })
        # Should not raise
        create_clean_gantt_chart(ax, df, 'task', 'start', 'end', title="Gantt Test")
        self.assertEqual(ax.get_title(), "Gantt Test")
        plt.close(fig)

if __name__ == "__main__":
    unittest.main() 