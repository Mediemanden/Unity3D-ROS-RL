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

namespace RosSharp.RosBridgeClient
{

    [RequireComponent(typeof(RosConnector))]
    public class VelocitySubscriber : MonoBehaviour
    {
        public GameObject UrdfModel;
        private VelocityManager velocityManager;
        private RosSocket rosSocket;
        public int UpdateTime = 1;
        public bool actionReceived = true;

        public void Start()
        {
            Time.timeScale = 1;
            rosSocket = transform.GetComponent<RosConnector>().RosSocket;
            rosSocket.Subscribe("/cmd_vel", "geometry_msgs/Twist", updateOdometry, UpdateTime);
            velocityManager = UrdfModel.GetComponent<VelocityManager>();
        }

        private void updateOdometry(Message message)
        {
            
            GeometryTwist geometryTwist= (GeometryTwist)message;
            Debug.Log("Action received");
            velocityManager.updateTransform(getLinearVelocity(geometryTwist), getAngularVelocity(geometryTwist));
            actionReceived = true;
        }

        private static Vector3 getLinearVelocity(GeometryTwist geometryTwist)
        {
            return new Vector3(
                -geometryTwist.linear.y,
                geometryTwist.linear.z,
                geometryTwist.linear.x);
        }

        private static Vector3 getAngularVelocity(GeometryTwist _geometryTwist)
        {
            return new Vector3(
                _geometryTwist.angular.x,
                -_geometryTwist.angular.z,
                _geometryTwist.angular.y);
        }

    }
}
