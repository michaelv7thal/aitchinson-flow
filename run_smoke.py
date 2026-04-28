import sys
import importlib.util

def run_script():
    sys.path.insert(0, 'scripts')
    
    from aitchinson_flow.config import Config
    
    original_init = Config.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.training_data.source = "llm_topk_probs"
        self.training.model_name = "per_token_bayesian_auditor"
        self.per_token_auditor.use_context = True
    
    Config.__init__ = new_init
    
    sys.argv = ["scripts/single_stage_train.py", "--smoke"]
    
    spec = importlib.util.spec_from_file_location("__main__", "scripts/single_stage_train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = module
    spec.loader.exec_module(module)

if __name__ == "__main__":
    run_script()
