# DVIS-DAQ Device Issue Analysis

## 🚨 **Root Cause of Device Issues**

The DVIS-DAQ model has **hardcoded CUDA references** throughout its codebase, making it extremely difficult to run on CPU. Here are the specific issues:

### **1. Hardcoded CUDA References Found:**

#### **In `track_module.py`:**
```python
# Line 264: Empty tensor creation
return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")

# Line 272: Empty tensor creation  
return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")

# Line 280: Empty tensor creation
return torch.empty(size=(0, ), dtype=torch.int64).to("cuda")

# Line 286: Boolean mask creation
disappear_fq_mask = torch.zeros(size=(fQ, ), dtype=torch.bool).to("cuda")

# Line 409: Tensor creation with fill value
fill_value=-1).to("cuda")

# Line 411: Boolean tensor creation
activated_queries_bool = torch.ones(size=(ms_outputs.shape[1], )).to("cuda") < 0

# Line 790: Mask thresholding
seg_mask = (mask[:, start:end, :, :].sigmoid() > 0.5).to("cuda")
```

#### **In `meta_architecture.py`:**
```python
# Line 554: Logits tensor creation
full_logits = torch.ones(num_frames, self.sem_seg_head.num_classes + 1).to(torch.float32).to("cuda") * -1e4

# Line 1317-1318: Padding masks and sequence IDs
padding_masks = torch.zeros(size=(1, 0, T), dtype=torch.bool).to("cuda")
seq_id_tensor = torch.IntTensor([]).to("cuda")

# Line 1324: Sequence ID tensor
seq_id_tensor = torch.IntTensor(seq_id_list).to("cuda")

# Line 1329: Top-k indices
topk_indices = torch.arange(scores.shape[0]).to("cuda")

# Line 1348: Naive padding masks
naive_padding_masks = torch.ones(size=(num_left, num_frames)).to("cuda") < 0
```

## 🔍 **Why This Happens:**

1. **Training-Optimized Code**: The model was designed for GPU training/inference
2. **Performance Assumptions**: Developers assumed CUDA would always be available
3. **Lazy Device Handling**: Instead of using `device` parameters, tensors are hardcoded to CUDA
4. **No CPU Fallback**: No consideration for CPU-only environments

## 🛠️ **Potential Solutions:**

### **Solution 1: Patch the Model Code (Recommended)**
Create a patched version that replaces hardcoded CUDA references with device-aware code:

```python
# Instead of:
tensor = torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")

# Use:
tensor = torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32, device=self.device)
```

### **Solution 2: Use a Different Model Architecture**
- Extract just the backbone for gradient analysis
- Use a simpler model that doesn't have hardcoded device references
- Focus on temporal features from earlier layers

### **Solution 3: Environment Variable Override**
Set environment variables to force CPU usage, though this may not work for all hardcoded references.

### **Solution 4: Gradient Extraction from Sub-components**
- Extract gradients from individual layers before they hit the problematic modules
- Focus on the backbone and early feature extraction layers
- Avoid the tracking and refinement modules that have the most CUDA dependencies

## 📊 **Assessment of CPU Feasibility:**

### **✅ What's Possible:**
- **Backbone layers**: Most backbone components can run on CPU
- **Early feature extraction**: Initial processing can be done on CPU
- **Simple gradient extraction**: Basic gradient computation works on CPU

### **❌ What's Problematic:**
- **Tracking modules**: Heavy CUDA dependencies
- **Refinement layers**: Complex temporal processing with hardcoded CUDA
- **Full model inference**: The complete pipeline has too many CUDA dependencies

## 🎯 **Recommended Approach:**

### **Option A: Patch the Model (Most Comprehensive)**
1. Create a patched version of the DVIS-DAQ model
2. Replace all hardcoded `.to("cuda")` with `.to(self.device)`
3. Add proper device handling throughout the codebase
4. Test thoroughly to ensure functionality is preserved

### **Option B: Extract Gradients from Backbone Only (Simpler)**
1. Load only the backbone and early layers
2. Extract gradients from these components
3. Focus on temporal features from early layers
4. Avoid the problematic tracking modules

### **Option C: Use Temporal Motion Analysis (Current Working Solution)**
1. Continue with the optical flow analysis
2. It's already providing meaningful insights
3. Focus on motion patterns rather than model gradients
4. Scale up to more videos and species

## 🔧 **Implementation Plan:**

### **For Option A (Patching):**
1. Create a script to automatically patch the model files
2. Replace all hardcoded CUDA references
3. Add device parameter handling
4. Test the patched model

### **For Option B (Backbone Only):**
1. Extract the backbone from the model
2. Create a simplified gradient extraction pipeline
3. Focus on temporal features from early layers
4. Compare with motion analysis results

## 📈 **Expected Outcomes:**

### **If Patching Works:**
- Full model gradient extraction on CPU
- True temporal gradient analysis
- Complete understanding of model behavior

### **If Backbone Extraction Works:**
- Partial model gradient analysis
- Focus on early temporal features
- Some insights into model behavior

### **If Neither Works:**
- Continue with temporal motion analysis
- It's already providing valuable insights
- Focus on motion patterns for species classification

## 🎯 **Next Steps:**

1. **Try Option B first** (backbone extraction) - simpler and faster
2. **If that fails, try Option A** (patching) - more comprehensive but complex
3. **Fall back to Option C** (temporal motion) - already working and valuable

The temporal motion analysis is already providing meaningful insights about species-specific motion patterns, so we have a working solution regardless of the model gradient issues.
