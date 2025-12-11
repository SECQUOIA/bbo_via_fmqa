"""
Integration tests for FMQA functionality
"""

import pytest
import numpy as np
import dimod

# Test that fmqa is importable
try:
    from fmqa import FMBQM
    FMQA_AVAILABLE = True
except ImportError:
    FMQA_AVAILABLE = False


@pytest.mark.skipif(not FMQA_AVAILABLE, reason="fmqa package not installed")
class TestFMBQMIntegration:
    """Test FMBQM integration"""
    
    def test_import_fmbqm(self):
        """Test that FMBQM can be imported"""
        from fmqa import FMBQM
        assert FMBQM is not None
    
    def test_create_from_data_binary(self):
        """Test creating FMBQM from binary data"""
        from fmqa import FMBQM
        
        # Create simple binary dataset
        x = np.array([[0, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0]], dtype=int)
        y = np.array([1.0, 2.0, 4.0, -8.0])
        
        # Create model
        model = FMBQM.from_data(x, y, num_epoch=10, learning_rate=1e-2)
        
        assert model is not None
        assert hasattr(model, 'fm')
    
    def test_sample_fmbqm(self):
        """Test sampling from FMBQM with simulated annealing"""
        from fmqa import FMBQM
        
        # Create simple dataset
        x = np.random.randint(2, size=(5, 8))
        y = np.random.randn(5)
        
        # Create and train model
        model = FMBQM.from_data(x, y, num_epoch=10, learning_rate=1e-2)
        
        # Sample using simulated annealing
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(model, num_reads=10)
        
        assert len(sampleset) > 0
        assert hasattr(sampleset, 'first')
    
    def test_train_and_predict(self):
        """Test training FMBQM and making predictions"""
        from fmqa import FMBQM
        
        # Create dataset where y = sum of x values
        np.random.seed(42)
        x = np.random.randint(2, size=(20, 4))
        y = x.sum(axis=1).astype(float)
        
        # Create model
        model = FMBQM.from_data(x, y, num_epoch=100, learning_rate=1e-2)
        
        # Sample solutions
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(model, num_reads=50)
        
        # Check that we got valid binary samples
        first_sample = sampleset.first.sample
        for key, val in first_sample.items():
            assert val in [0, 1]
    
    def test_vartype_detection(self):
        """Test that FMBQM correctly detects variable type"""
        from fmqa import FMBQM
        
        # Binary data
        x_binary = np.array([[0, 1, 0], [1, 0, 1]])
        y = np.array([0.5, -0.5])
        
        model = FMBQM.from_data(x_binary, y, num_epoch=10)
        assert model.vartype.name == 'BINARY'
    
    def test_multiple_training_cycles(self):
        """Test multiple training cycles (online learning)"""
        from fmqa import FMBQM
        
        # Initial dataset
        x = np.random.randint(2, size=(5, 6))
        y = np.random.randn(5)
        
        # Create initial model
        model = FMBQM.from_data(x, y, num_epoch=10)
        
        # Add more data and retrain
        x_new = np.random.randint(2, size=(3, 6))
        y_new = np.random.randn(3)
        
        x_combined = np.vstack([x, x_new])
        y_combined = np.hstack([y, y_new])
        
        model.train(x_combined, y_combined, num_epoch=10)
        
        # Should still be able to sample
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(model, num_reads=5)
        assert len(sampleset) > 0


class TestDimodIntegration:
    """Test dimod integration"""
    
    def test_simulated_annealing_available(self):
        """Test that SimulatedAnnealingSampler is available"""
        sampler = dimod.SimulatedAnnealingSampler()
        assert sampler is not None
    
    def test_sample_simple_bqm(self):
        """Test sampling a simple BQM"""
        # Create a simple BQM: minimize x0 + x1
        bqm = dimod.BinaryQuadraticModel({0: 1, 1: 1}, {}, 0.0, 'BINARY')
        
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=10)
        
        # Optimal solution should be all zeros
        best = sampleset.first
        assert best.energy <= 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
