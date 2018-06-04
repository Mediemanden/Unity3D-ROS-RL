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

namespace RosSharp
{
    public class VelocityManager : MonoBehaviour
    {
        private Vector3 linearVelocity;
        private Vector3 angularVelocity;
        private bool doUpdate;
        private Rigidbody rigid;
        public WheelCollider leftWheel, rightWheel, frontWheel, backWheel;

        private float targetVelocityMagnitude = 0.5f;

        private void Start()
        {
            rigid = gameObject.GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            if (doUpdate)
            {
                
                leftWheel.motorTorque = (linearVelocity.z + angularVelocity.y) * 100;
                rightWheel.motorTorque = (linearVelocity.z + -1*angularVelocity.y) * 100;
            }
            //Debug.Log("Current speed: " + rigid.velocity.magnitude);
        }

        public void updateTransform(Vector3 _linearVelocity, Vector3 _angularVelocity)
        {
            linearVelocity = _linearVelocity;
            angularVelocity = _angularVelocity;
            linearVelocity.Normalize();
            angularVelocity.Normalize();
            doUpdate = true;
        }
    }
}
