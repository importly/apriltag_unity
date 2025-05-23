using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class ApriltagCameraController : MonoBehaviour
{
    [Tooltip("Port to listen on for UDP AprilTag poses")]
    public int listenPort = 5065;

    [Tooltip("The _Camera_ GameObject to move/rotate")]
    public Transform cameraObject;

    [Tooltip("How aggressively to follow the raw pose.\n0 = max smoothing (very laggy), 1 = no smoothing (jerky)")]
    [Range(0f, 1f)]
    public float smoothingFactor = 0.1f;

    private IPEndPoint remoteEP;

    // Smoothed pose state:
    private Vector3 smoothPos;
    private Quaternion smoothRot;
    private UdpClient udpClient;

    private void Start()
    {
        udpClient = new UdpClient(listenPort);
        udpClient.Client.Blocking = false;
        remoteEP = new IPEndPoint(IPAddress.Any, listenPort);

        // Initialize the smoothed state so it doesn't jump on the first frame:
        if (cameraObject != null)
        {
            smoothPos = cameraObject.localPosition;
            smoothRot = cameraObject.localRotation;
        }
        else
        {
            smoothPos = Vector3.zero;
            smoothRot = Quaternion.identity;
        }
    }

    private void Update()
    {
        if (cameraObject == null) return;

        // Process all pending packets
        while (udpClient.Available > 0)
            try
            {
                var data = udpClient.Receive(ref remoteEP);
                var msg = Encoding.UTF8.GetString(data);
                var parts = msg.Split(',');
                if (parts.Length < 7)
                    continue;

                // --- parse tag‐in‐camera pose ---
                var tx = float.Parse(parts[1]);
                var ty = float.Parse(parts[2]);
                var tz = float.Parse(parts[3]);
                var t_tagInCam = new Vector3(tx, ty, tz);

                var rx = float.Parse(parts[4]);
                var ry = float.Parse(parts[5]);
                var rz = float.Parse(parts[6]);
                var rod = new Vector3(rx, ry, rz);

                var angleRad = rod.magnitude;
                var axis = angleRad > 1e-6f ? rod / angleRad : Vector3.up;
                var angleDeg = angleRad * Mathf.Rad2Deg;
                var q_camToTag = Quaternion.AngleAxis(angleDeg, axis);

                // invert to get tag→camera
                var q_tagToCam = Quaternion.Inverse(q_camToTag);
                var t_tagToCam = -(q_tagToCam * t_tagInCam);

                // Unity camera pose
                var camPos = new Vector3(
                    -t_tagToCam.x,
                    t_tagToCam.y,
                    t_tagToCam.z
                );
                var euler = q_tagToCam.eulerAngles;
                var camRot = Quaternion.Euler(
                    euler.x,
                    -euler.y,
                    -euler.z
                );

                // --- apply exponential smoothing ---
                smoothPos = Vector3.Lerp(smoothPos, camPos, smoothingFactor);
                smoothRot = Quaternion.Slerp(smoothRot, camRot, smoothingFactor);

                cameraObject.localPosition = smoothPos;
                cameraObject.localRotation = smoothRot;
            }
            catch (SocketException)
            {
                // no more packets
            }
    }

    private void OnApplicationQuit()
    {
        udpClient.Close();
    }
}