using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class ApriltagReceiver : MonoBehaviour
{
    [Tooltip("Port to listen on for UDP AprilTag poses")]
    public int listenPort = 5065;

    [Tooltip("The GameObject to move/rotate")]
    public Transform tagObject;

    UdpClient udpClient;
    IPEndPoint remoteEP;
    Vector3 lastPos = Vector3.zero;
    Quaternion lastRot = Quaternion.identity;

    void Start()
    {
        udpClient = new UdpClient(listenPort);
        udpClient.Client.Blocking = false;  // non-blocking
        remoteEP = new IPEndPoint(IPAddress.Any, listenPort);
    }

    void Update()
    {
        // read all waiting packets, use the last one
        while (udpClient.Available > 0)
        {
            try
            {
                byte[] data = udpClient.Receive(ref remoteEP);
                string msg = Encoding.UTF8.GetString(data);
                // format: tagID,tx,ty,tz,rx,ry,rz
                var parts = msg.Split(',');
                if (parts.Length >= 7)
                {
                    // parse position
                    float tx = float.Parse(parts[1]);
                    float ty = float.Parse(parts[2]);
                    float tz = float.Parse(parts[3]);
                    lastPos = new Vector3(tx, ty, tz);

                    // parse rotation-vector (radians)
                    float rx = float.Parse(parts[4]);
                    float ry = float.Parse(parts[5]);
                    float rz = float.Parse(parts[6]);
                    Vector3 axisAngle = new Vector3(rx, ry, rz);
                    float angle = axisAngle.magnitude * Mathf.Rad2Deg;
                    if (axisAngle.sqrMagnitude > 1e-6f)
                        lastRot = Quaternion.AngleAxis(angle, axisAngle.normalized);
                    else
                        lastRot = Quaternion.identity;
                }
            }
            catch (SocketException) { /* no data */ }
        }

        // apply to your GameObject
        if (tagObject != null)
        {
            tagObject.localPosition = lastPos;
            tagObject.localRotation = lastRot;
        }
    }

    void OnApplicationQuit()
    {
        udpClient.Close();
    }
}
