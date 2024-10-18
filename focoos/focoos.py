from enum import Enum
from pathlib import Path
import time
from typing import Union
from supervision import Detections
import requests
import os
from focoos.utils.logger import setup_logging,get_logger


class DeploymentMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"

class FocoosModel:
    def __init__(self, model_ref:str, api_key:str):
        pass
    def infer(self, image_path,threshold:float=0.5) -> Detections:
        pass
    def benchmark(self, image_path) :
        pass

class RemoteModel(FocoosModel):
    def __init__(self, model_name:str,api_key:str,host_url:str):
        self.model_name = model_name
        self.host_url = host_url
        self.api_key = api_key
        self.default_headers = {"Authorization": f"Bearer {self.api_key}", "user_agent": "focoos/0.0.1"}
        self.logger = get_logger()
        self.logger.info(f"Initialized RemoteModel: {self.model_name}")


    


    def infer(self, image_path:Union[str,Path],threshold:float=0.5) -> Detections:
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")
        files = {"file": open(image_path, "rb")}
        res = requests.post(f"{self.host_url}/models/{self.model_name}/inference?confidence_threshold={threshold}", headers=self.default_headers, files=files)
        if res.status_code == 200:  
            return res.json()
        else:
            self.logger.error(f"Failed to infer: {res.status_code} {res.text}")
            raise ValueError(f"Failed to infer: {res.status_code} {res.text}")
        
    

class LocalModel(FocoosModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def infer(self, image_path) -> Detections:
        pass
       # return model.infer(image_path)


class Focoos:
    def __init__(self, api_key:str, host_url:str="https://api.dev.focoos.ai/v0"):
        self.logger = setup_logging()
        self.api_key = api_key
        if not self.api_key:
            self.logger.error("API key is required 🤖")
            raise ValueError("API key is required 🤖")
        
        self.host_url = host_url
        self.default_headers = {"Authorization": f"Bearer {self.api_key}", "user_agent": "focoos/0.0.1"}
        self.user_info = self._get_user_info()
        self.logger.info(f"Currently logged as: {self.user_info['email']}")

    def _get_user_info(self):
        res = requests.get(f"{self.host_url}/user/", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to get user info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get user info: {res.status_code} {res.text}")
        
    def get_model_info(self,model_name:str):
        res = requests.get(f"{self.host_url}/models/{model_name}", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to get model info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get model info: {res.status_code} {res.text}") 
    def list_models(self):
        res = requests.get(f"{self.host_url}/models/", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to list models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list models: {res.status_code} {res.text}")
    def list_focoos_models(self):
        res = requests.get(f"{self.host_url}/models/focoos-models", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to list focoos models: {res.status_code} {res.text}")
            raise ValueError(f"Failed to list focoos models: {res.status_code} {res.text}")

    def unload_model(self, model_name:str):
        res = requests.delete(f"{self.host_url}/models/{model_name}/deploy", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to unload model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to unload model: {res.status_code} {res.text}")

    def get_model(self, model_ref:str, deployment_mode:DeploymentMode=DeploymentMode.LOCAL)-> FocoosModel:
        model_info = self.get_model_info(model_ref)
        status = model_info['status']
        self.logger.info(f"Model status: {status}")
        if deployment_mode == DeploymentMode.LOCAL:
            raise NotImplementedError("Local deployment is not implemented yet 🤖")
        elif deployment_mode == DeploymentMode.REMOTE:
            deployment_status = self._deployment_info(model_ref)["status"]
            self.logger.info(f"🤖 Deployment status: {deployment_status}")
            if deployment_status == "READY":
                return RemoteModel(model_ref, self.api_key, self.host_url)
            else:
                self.logger.info(f"🚀 Deploying model {model_ref} to inference endpoint... this might take a while.")
                res = self._deploy_model(model_ref)

                for i in range(10):
                    deployment_status = self._deployment_info(model_ref)["status"]
                    if deployment_status == "READY":
                        break
                    self.logger.info(f"Waiting for model {model_ref} to be ready ... {i}")
                    time.sleep(1+i)
                self.logger.info(f"✨ Model {model_ref} deployed successfull: {res}")
                return RemoteModel(model_ref, self.api_key, self.host_url)
                
    
    def _deployment_info(self, model_name:str):
        res = requests.get(f"{self.host_url}/models/{model_name}/deploy", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to get deployment info: {res.status_code} {res.text}")
            raise ValueError(f"Failed to get deployment info: {res.status_code} {res.text}")


    def _deploy_model(self, model_name:str):
        res = requests.post(f"{self.host_url}/models/{model_name}/deploy", headers=self.default_headers)
        if res.status_code == 200:
            return res.json()
        else:
            self.logger.error(f"Failed to deploy model: {res.status_code} {res.text}")
            raise ValueError(f"Failed to deploy model: {res.status_code} {res.text}")