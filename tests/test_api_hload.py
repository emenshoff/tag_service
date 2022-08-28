
import json
import unittest
import aiohttp
import asyncio
import logging

from tests.config import TEST_PROFILE, TEST_HOST, TEST_PORT, TEST_LOG_LEVEL, TEST_API_VERSION

log = logging.getLogger(__name__)
log.setLevel(TEST_LOG_LEVEL)

PROFILE = TEST_PROFILE
MAX_SESSIONS = 30
URL = f"http://{TEST_HOST}:{TEST_PORT}/api"

fail_count = 0
passed_count = 0

PAYLOAD = {
    
        "content_prefix": "/media/nvme1tb/datasets/aromam/media_old/",
        "content_type":  "file",           
        "get_original_image": False,
        "get_item_image": False,
        "version": TEST_API_VERSION,
        "profile": PROFILE,
        "content": [
            "798972b0738116811195508ab060e02c.jpg",
            "3d9425c56b3d4320323b345e3302d987.jpg", 
            "e38fb79050804f4b97694981b432b3f5.jpg",
            "7281e800a04adad3b672b529416bbc20.jpg",
            "1ba37285c2cbdb5e9bc9ea8f0bcb9edd.jpg",
            "d73c010da65c88509fcff3daa408ba80.jpg",
            "ecad2477be994569e0ced1ee70ce2781.jpg",
            "89021628fb9989a6bc0267d4484cb3f8.jpg",
            "a7e46ad4614f50afeb1332dd39153d21.jpg",
            "3d6f24e9eab2f5edd071557aa68d5ee6.jpg",
            "855cf79dd45030695d689124bc7c6ee4.jpg"
        ]
}

async def request_task(task_id, sem, payload=PAYLOAD, url=URL):
    global fail_count, passed_count
    try:
        async with sem:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=json.dumps(payload)) as response:    
                    # log.info(session.headers)            
                    if response.status == 200:
                        try:
                            content = await response.read()      
                            if content is not None:
                                content = json.loads(content)
                                
                                err_msg = content.get("error") or ""
                                if err_msg == "":
                                    passed_count += 1
                                    log.info(f"{task_id} prediction Ok. response: {content}")
                                else:
                                    fail_count += 1
                                    log.info(f"{task_id} prediction failed: {content}")
                            
                        except Exception as ex:
                            fail_count += 1
                            err_msg = f"{task_id} Read error: {ex}"
                            raise Exception(err_msg)
                    else:
                        fail_count += 1
                        err_msg = f"{task_id} HTTP error, http response code: {response.status}"
                        raise Exception(err_msg)
                            
    except Exception as ex:
        err_msg = f"Request fail: {url}, error: {ex}"  #, exc_info=True
        log.error(err_msg)

       
# async content loader
async def run_tasks(num_tasks=MAX_SESSIONS):
    sem = asyncio.Semaphore(MAX_SESSIONS)
    tasks = [request_task(i, sem) for i in range(num_tasks)]
    await asyncio.gather(*tasks)


class TestHighLoad(unittest.TestCase):   
    
    def test_api_version(self):
        asyncio.run(run_tasks())
        log.info(f"failed tasks: {fail_count}")        
        self.assertEqual(fail_count, 0)


   



# if __name__ == "__main__":
#     unittest.main()