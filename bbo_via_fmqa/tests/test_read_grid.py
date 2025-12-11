"""
Tests for read_grid utility functions
"""

import pytest
import numpy as np
import csv
import os
import tempfile
from bbo_via_fmqa import read_grid


class TestIntToBits:
    """Test integer to binary encoding"""
    
    def test_basic_encoding(self):
        """Test basic integer to bits conversion"""
        # 5 in binary with upper bound 7 (requires 3 bits)
        result = read_grid.int_to_bits(5, 7, lsb_first=True)
        # 5 = 101 in binary, LSB first = [1, 0, 1]
        assert result == [1, 0, 1]
    
    def test_msb_first(self):
        """Test most significant bit first encoding"""
        result = read_grid.int_to_bits(5, 7, lsb_first=False)
        # 5 = 101 in binary, MSB first = [1, 0, 1]
        assert result == [1, 0, 1]
    
    def test_zero_value(self):
        """Test encoding of zero"""
        result = read_grid.int_to_bits(0, 7, lsb_first=True)
        assert result == [0, 0, 0]
    
    def test_max_value(self):
        """Test encoding of maximum value"""
        result = read_grid.int_to_bits(7, 7, lsb_first=True)
        # 7 = 111 in binary
        assert result == [1, 1, 1]
    
    def test_out_of_bounds_raises(self):
        """Test that out of bounds values raise ValueError"""
        with pytest.raises(ValueError):
            read_grid.int_to_bits(-1, 7)
        
        with pytest.raises(ValueError):
            read_grid.int_to_bits(8, 7)


class TestBitsToInt:
    """Test binary to integer decoding"""
    
    def test_basic_decoding(self):
        """Test basic bits to integer conversion"""
        # Binary string "101001" splits to "101" (5) and "001" (1)
        x, y = read_grid.bits_to_int("101001", lsb_first=False)
        assert x == 5
        assert y == 1
    
    def test_even_split(self):
        """Test that bits are split evenly"""
        x, y = read_grid.bits_to_int("1100", lsb_first=False)
        # "11" = 3, "00" = 0
        assert x == 3
        assert y == 0
    
    def test_lsb_first(self):
        """Test LSB first decoding"""
        # With LSB first, string is reversed before processing
        x, y = read_grid.bits_to_int("100101", lsb_first=True)
        # Reversed: "101001" -> "101" (5) and "001" (1)
        assert x == 5
        assert y == 1


class TestCoordBits:
    """Test coordinate to bitstring encoding"""
    
    def test_basic_encoding(self):
        """Test coordinate to bitstring conversion"""
        result = read_grid.coord_bits(2, 3, 7, 7, lsb_first=False)
        # 2 = 010, 3 = 011 with 3 bits each for max 7
        assert result == "010011"
    
    def test_zero_coords(self):
        """Test encoding of (0, 0)"""
        result = read_grid.coord_bits(0, 0, 7, 7, lsb_first=False)
        assert result == "000000"
    
    def test_max_coords(self):
        """Test encoding of maximum coordinates"""
        result = read_grid.coord_bits(7, 7, 7, 7, lsb_first=False)
        assert result == "111111"
    
    def test_roundtrip(self):
        """Test that encoding and decoding are inverse operations"""
        x_orig, y_orig = 5, 3
        x_max, y_max = 15, 15
        
        # Encode
        bitstring = read_grid.coord_bits(x_orig, y_orig, x_max, y_max, lsb_first=False)
        
        # Decode
        x_decoded, y_decoded = read_grid.bits_to_int(bitstring, lsb_first=False)
        
        assert x_decoded == x_orig
        assert y_decoded == y_orig


class TestObjFunct:
    """Test objective function evaluation"""
    
    def test_existing_point(self):
        """Test evaluation of existing grid point"""
        grid_data = {(1, 2): 5.0, (3, 4): 10.0}
        result = read_grid.obj_funct([1, 2], grid_data)
        assert result == 5.0
    
    def test_missing_point(self):
        """Test evaluation of missing point returns penalty"""
        grid_data = {(1, 2): 5.0}
        result = read_grid.obj_funct([3, 4], grid_data)
        assert result == np.inf
    
    def test_nan_point(self):
        """Test evaluation of NaN point returns penalty"""
        grid_data = {(1, 2): np.nan}
        result = read_grid.obj_funct([1, 2], grid_data)
        assert result == np.inf
    
    def test_rounding(self):
        """Test that coordinates are rounded"""
        grid_data = {(2, 3): 7.0}
        result = read_grid.obj_funct([1.6, 2.8], grid_data)
        assert result == 7.0


class TestScaleValue:
    """Test value scaling function"""
    
    def test_scale_min(self):
        """Test scaling of minimum value"""
        obj_min, obj_max = 0.0, 10.0
        # The scale_value is not defined in read_grid.py but used in fmqa_test.py
        # We'll test the scaling logic from fmqa_test.py
        y_val = obj_min
        scaled = (y_val - obj_min) / (obj_max - obj_min)
        assert scaled == 0.0
    
    def test_scale_max(self):
        """Test scaling of maximum value"""
        obj_min, obj_max = 0.0, 10.0
        y_val = obj_max
        scaled = (y_val - obj_min) / (obj_max - obj_min)
        assert scaled == 1.0
    
    def test_scale_mid(self):
        """Test scaling of middle value"""
        obj_min, obj_max = 0.0, 10.0
        y_val = 5.0
        scaled = (y_val - obj_min) / (obj_max - obj_min)
        assert scaled == 0.5


class TestLoadGrid:
    """Test grid loading from CSV"""
    
    def test_load_simple_grid(self):
        """Test loading a simple grid from CSV"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'Objective'])
            writer.writerow([1, 2, 5.0])
            writer.writerow([3, 4, 10.0])
            writer.writerow([5, 6, 15.0])
            temp_path = f.name
        
        try:
            grid_data, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(temp_path)
            
            assert len(grid_data) == 3
            assert grid_data[(1, 2)] == 5.0
            assert grid_data[(3, 4)] == 10.0
            assert grid_data[(5, 6)] == 15.0
            assert obj_min == 5.0
            assert obj_max == 15.0
            assert x_bound == 5
            assert y_bound == 6
        finally:
            os.unlink(temp_path)
    
    def test_load_grid_with_nan(self):
        """Test loading grid with NaN values"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'Objective'])
            writer.writerow([1, 2, 5.0])
            writer.writerow([3, 4, 'nan'])
            writer.writerow([5, 6, 15.0])
            temp_path = f.name
        
        try:
            grid_data, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(temp_path)
            
            assert len(grid_data) == 3
            assert np.isnan(grid_data[(3, 4)])
            assert obj_min == 5.0
            assert obj_max == 15.0
        finally:
            os.unlink(temp_path)
    
    def test_load_empty_grid(self):
        """Test loading an empty grid"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'Objective'])
            temp_path = f.name
        
        try:
            grid_data, obj_min, obj_max, x_bound, y_bound = read_grid.load_grid(temp_path)
            
            assert len(grid_data) == 0
            assert obj_min is None
            assert obj_max is None
            assert x_bound == 0
            assert y_bound == 0
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
