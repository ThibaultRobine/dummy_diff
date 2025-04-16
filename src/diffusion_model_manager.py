# diffusion_model_manager.py

trained_diffusion_model = None

def set_trained_diffusion_model(model,diffusion):
    global trained_diffusion_model
    trained_diffusion_model = model, diffusion

def get_trained_diffusion_model():
    return trained_diffusion_model
