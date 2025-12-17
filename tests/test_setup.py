from llama_cpp import Llama
# n_gpu_layers=1 или больше активирует CUDA
llm = Llama(model_path="path_to_model.gguf", n_gpu_layers=20)
print("CUDA is available:", llm.has_cuda)