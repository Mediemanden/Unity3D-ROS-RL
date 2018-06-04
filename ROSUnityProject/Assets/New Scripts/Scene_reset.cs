/*
© Siemens AG, 2017
Author: Dr. Martin Bischoff (martin.bischoff@siemens.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/



using UnityEngine;
using UnityEngine.SceneManagement;

namespace RosSharp.RosBridgeClient
{

    [RequireComponent(typeof(RosConnector))]
    public class Scene_reset : MonoBehaviour
    {
        private RosConnector rosConnector;
        private RosSocket rosSocket;
        public int UpdateTime = 1;
        private Scene myScene;
        private StandardString doneString = new StandardString();

        public void Start()
        {
            rosConnector = gameObject.GetComponent<RosConnector>();
            myScene = SceneManager.GetActiveScene();
            rosSocket = transform.GetComponent<RosConnector>().RosSocket;
            rosSocket.Subscribe("done", "std_msgs/String", sceneReset, UpdateTime);
            doneString.data = "False";
        }

        public void Update()
        {
            if (doneString.data == "True")
            {
                rosConnector.Disconnect();
                SceneManager.LoadScene(myScene.name);
                doneString.data = "False";
            }
        }

        private void sceneReset(Message message)
        {
            doneString = (StandardString)message;
            Debug.Log(doneString.data);

        }
    }
}