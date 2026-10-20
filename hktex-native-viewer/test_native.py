"""CPU tests of offline ring state restoration, ID colours and image composition."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from render_native import ring_mode, palette, contact_sheet, build_html, KINDS


class NativeTests(unittest.TestCase):
    def test_ring_state_restored_even_on_failure(self):
        model = torch.nn.Module()
        model._kernel_colours = torch.nn.Parameter(torch.tensor([[.2, .3, .4], [-.1, .6, .2]]))
        model._mean_colour = torch.nn.Parameter(torch.tensor([[.5, .5, .5]]))
        model.kernel_filter_func = lambda x: x
        original = (model._kernel_colours, model._mean_colour, model.kernel_filter_func)
        values = [p.detach().clone() for p in original[:2]]
        rng = torch.random.get_rng_state().clone()
        def box(x, epsilon, sharpness=None, thickness=.03):
            return ((x >= epsilon-thickness) & (x < epsilon)).float()
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            with ring_mode(model, [7, 19], box, .03):
                np.testing.assert_array_equal(model._kernel_colours.detach().numpy(), palette([7, 19]))
                self.assertEqual(float(model._mean_colour.abs().sum()), 0.)
                result = model.kernel_filter_func(torch.tensor([.46, .49, .51]), epsilon=.5)
                torch.testing.assert_close(result, torch.tensor([0., 1., 0.]))
                raise RuntimeError('interrupted')
        self.assertIs(model._kernel_colours, original[0])
        self.assertIs(model._mean_colour, original[1])
        self.assertIs(model.kernel_filter_func, original[2])
        for restored, saved in zip(original, values):
            torch.testing.assert_close(restored, saved)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)

    def test_palette_follows_ids_after_prune_or_reorder(self):
        full = palette([1, 90, 52, 700])
        subset = palette([700, 90])
        np.testing.assert_array_equal(subset, full[[3, 1]])

    def test_contact_sheet_has_native_pixels_without_resizing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            size = 64
            colours = [[250, 10, 20], [30, 240, 40], [50, 60, 230]]
            for i in range(4):
                for key, colour in zip(KINDS, colours):
                    for suffix in ['', '_raw']:
                        Image.new('RGB', (size, size), tuple(colour)).save(root/f'{key}_{i:03d}{suffix}.png')
            contact_sheet(root, 4, size)
            contact_sheet(root, 4, size, raw=True)
            with Image.open(root/'preview_views.png') as sheet:
                self.assertEqual(sheet.size, (3*size, 4*(size+28)))
                for col, colour in enumerate(colours):
                    self.assertEqual(sheet.getpixel((col*size+5, 28+5)), tuple(colour))
            build_html(root, [dict(directory='snapshot', label='Final', video=False)], 4, True)
            page = (root/'index.html').read_text()
            self.assertNotIn('PAYLOAD', page)
            self.assertNotIn('FRAMES', page)
            self.assertIn('preview_views_raw.png', page)


if __name__ == '__main__':
    unittest.main()
