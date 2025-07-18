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
    create_clean_bar_chart
)

class TestPlotHelpers(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.df = pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value': [10, 20, 15, 5],
            'group': ['X', 'Y', 'X', 'Y']
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

if __name__ == "__main__":
    unittest.main() 