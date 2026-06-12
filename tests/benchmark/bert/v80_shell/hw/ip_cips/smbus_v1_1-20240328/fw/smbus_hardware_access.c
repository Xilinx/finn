/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains functions to read and write to the SMBus IP register map
 * It also has functions for writing Descriptors to the IPs Descriptor FIFO
 *
 * @file smbus_hardware_access.c
 *
 */

#include "smbus.h"
#include "smbus_internal.h"
#include "smbus_hardware_access.h"

char* pDescriptorTargetRead             = "DESC_TARGET_READ";
char* pDescriptorTargetPEC              = "DESC_TARGET_READ_PEC";
char* pDescriptorTargetWriteACK         = "DESC_TARGET_WRITE_ACK";
char* pDescriptorTargetWriteNACK        = "DESC_TARGET_WRITE_NACK";
char* pDescriptorTargetWritePEC         = "DESC_TARGET_WRITE_PEC";
char* pDescriptorControllerReadStart    = "DESC_CONTROLLER_READ_START";
char* pDescriptorControllerReadQuick    = "DESC_CONTROLLER_READ_QUICK";
char* pDescriptorControllerReadByte     = "DESC_CONTROLLER_READ_BYTE";
char* pDescriptorControllerReadStop     = "DESC_CONTROLLER_READ_STOP";
char* pDescriptorControllerReadPEC      = "DESC_CONTROLLER_READ_PEC";
char* pDescriptorControllerWriteStart   = "DESC_CONTROLLER_WRITE_START";
char* pDescriptorControllerWriteQuick   = "DESC_CONTROLLER_WRITE_QUICK";
char* pDescriptorControllerWriteByte    = "DESC_CONTROLLER_WRITE_BYTE";
char* pDescriptorControllerWriteStop    = "DESC_CONTROLLER_WRITE_STOP";
char* pDescriptorControllerWritePEC     = "DESC_CONTROLLER_WRITE_PEC";
char* pDescriptorUnknown                = "DESC_UNKNOWN";

/********************** Static function declarations ***************************/

/******************************************************************************
*
* @brief    Reads the uint32_t value from a hardware adddress
*
* @param    pvAddr is a pointer to a hardware address
*
* @return   uint32_t
*
* @note     None.
*
*****************************************************************************/
static inline uint32_t prvulSMBusIn32( void* pvAddr );

/******************************************************************************
*
* @brief    Writes a uint32_t value to a hardware adddress
*
* @param    pvAddr is a pointer to a hardware address
* @param    ulValue is uint32_t value to be written
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static inline void prvvSMBusOut32( void* pvAddr, uint32_t ulValue );

/******************************************************************************
*
* @brief    Look up the descriptor uint8_t value for the given descriptor ENUM
*
* @param    xDescriptor is the descriptor ENUM to use for the look up.
*
* @return   uint8_t descriptor ID value
*
* @note     None.
*
*****************************************************************************/
static uint8_t prvucSMBusGenericDescriptorIDLookup( SMBus_HW_Descriptor_Type xDescriptor );

/******************************************************************************
*
* @brief    Given the descriptor, look up the descriptor ID and write it into the
*           IP's Target descriptor FIFO. Depending on the descriptor type a payload
*           byte may be written along with the descriptor.
*           The ucNoStatusCheck can be used to check if the FIFO is full before
*           writing the descriptor or ignore the check
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    xDescriptor is the descriptor ENUM
* @param    ucPayload is the optional payload byte
* @param    ucNoStatusCheck is a boolean check
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static uint8_t prvucSMBusTargetDescriptorApply( SMBUS_PROFILE_TYPE* pxSMBusProfile, SMBus_HW_Descriptor_Type xDescriptor,
                                                uint8_t ucPayload, uint8_t ucNoStatusCheck );

/******************************************************************************
*
* @brief    Given the descriptor, look up the descriptor ID and write it into the
*           IP's Controller descriptor FIFO. Depending on the descriptor type a
*           payload byte may be written along with the descriptor.
*           The ucNoStatusCheck can be used to check if the FIFO is full before
*           writing the descriptor or ignore the check
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    xDescriptor is the descriptor ENUM
* @param    ucPayload is the optional payload byte
* @param    ucNoStatusCheck is a boolean check
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static uint8_t prvucSMBusControllerDescriptorApply( SMBUS_PROFILE_TYPE* pxSMBusProfile,
                                                    SMBus_HW_Descriptor_Type xDescriptor, uint8_t ucPayload,
                                                    uint8_t ucNoStatusCheck );

/******************************************************************************
*
* @brief    Return the value read form the register offset location
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulRegisterOffset offset from the start of the IP's base address
*
* @return   uint32_t value
*
* @note     None.
*
*****************************************************************************/
static uint32_t prvulSMBusHardwareRead( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulRegisterOffset );

/******************************************************************************
*
* @brief    Write the supplied value to the register location
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulRegisterOffset offset from the start of the IP's base address
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHardwareWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulRegisterOffset, uint32_t ulValue );

/******************************************************************************
*
* @brief    Modify the masked part of the register with the value supplied
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulRegisterOffset offset from the start of the IP's base address
* @param    ulMask bitmask to use for the write
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHardwareWriteWithMask( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulRegisterOffset,
                                    uint32_t ulMask, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_0
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl0Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_0
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl0Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_1
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl1Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_1
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl1Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_2
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl2Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_2
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl2Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_3
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl3Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_3
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl3Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_4
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl4Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_4
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl4Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_5
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl5Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_5
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl5Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_6
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl6Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_6
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl6Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_7
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl7Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_7
*           register
*
* @param    pxSMBusProfile is a pointer to the SMBus profile.
* @param    ulValue is the value to write
*
* @return   None
*
* @note     None.
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl7Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue );

/*******************************************************************************/

/******************************************************************************
*
* @brief    Reads the uint32_t value from a hardware adddress
*
*****************************************************************************/
static inline uint32_t prvulSMBusIn32( void* pvAddr )
{
    return *( volatile uint32_t* )pvAddr;
}

/******************************************************************************
*
* @brief    Writes a uint32_t value to a hardware adddress
*
*****************************************************************************/
static inline void prvvSMBusOut32( void* pvAddr, uint32_t ulValue )
{
    /* write 32 bit value to specified address */
    volatile uint32_t* LocalAddr = ( volatile uint32_t* )pvAddr;
    *LocalAddr = ulValue;
}

/******************************************************************************
*
* @brief    Look up the descriptor uint8_t value for the given descriptor ENUM
*
*****************************************************************************/
static uint8_t prvucSMBusGenericDescriptorIDLookup( SMBus_HW_Descriptor_Type xDescriptor )
{
    uint8_t ucDescriptorID = 0xFF;

    switch( xDescriptor )
    {
    case DESC_CONTROLLER_WRITE_START:
        ucDescriptorID = 0x0;
        break;
    
    case DESC_CONTROLLER_WRITE_QUICK:
        ucDescriptorID = 0x1;
        break;
    
    case DESC_CONTROLLER_WRITE_BYTE:
        ucDescriptorID = 0x2;
        break;
    
    case DESC_CONTROLLER_WRITE_STOP:
        ucDescriptorID = 0x3;
        break;
    
    case DESC_CONTROLLER_WRITE_PEC:
        ucDescriptorID = 0x4;
        break;
    
    case DESC_CONTROLLER_READ_START:
        ucDescriptorID = 0x8;
        break;
    
    case DESC_CONTROLLER_READ_QUICK:
        ucDescriptorID = 0x9;
        break;
    
    case DESC_CONTROLLER_READ_BYTE:
        ucDescriptorID = 0xA;
        break;
    
    case DESC_CONTROLLER_READ_STOP:
        ucDescriptorID = 0xB;
        break;
    
    case DESC_CONTROLLER_READ_PEC:
        ucDescriptorID = 0xC;
        break;    
    
    case DESC_TARGET_WRITE_ACK:
        ucDescriptorID = 0x0;
        break;
    
    case DESC_TARGET_WRITE_NACK:
        ucDescriptorID = 0x1;
        break;
    
    case DESC_TARGET_WRITE_PEC:
        ucDescriptorID = 0x2;
        break;
    
    case DESC_TARGET_READ:
        ucDescriptorID = 0x8;
        break;
    
    case DESC_TARGET_READ_PEC:
        ucDescriptorID = 0x9;
        break;
    
    default:
        break;
    }

    return ucDescriptorID;
}

/******************************************************************************
*
* @brief    Given the descriptor, look up the descriptor ID and write it into the
*           IP's Target descriptor FIFO. Depending on the descriptor type a payload
*           byte may be written along with the descriptor.
*           The ucNoStatusCheck can be used to check if the FIFO is full before
*           writing the descriptor or ignore the check
*
*****************************************************************************/
static uint8_t prvucSMBusTargetDescriptorApply( SMBUS_PROFILE_TYPE* pxSMBusProfile, SMBus_HW_Descriptor_Type xDescriptor,
                                                uint8_t ucPayload, uint8_t ucNoStatusCheck )
{
    uint8_t  ucReturnCode   = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    uint8_t  ucDescriptorID = 0;
    uint32_t ulValue        = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( DESC_MAX > xDescriptor ) )
    {
        ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS;

        if( SMBUS_FALSE == ucNoStatusCheck )
        {
            if( SMBUS_DESC_FIFO_IS_FULL == ulSMBusHWReadTgtDescStatusFull( pxSMBusProfile ) )
            {
                ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
            }
        }

        if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS == ucReturnCode )
        {
            ucDescriptorID = prvucSMBusGenericDescriptorIDLookup( xDescriptor );
            ulValue = ( ( uint32_t )ucDescriptorID << SMBUS_TGT_DESC_FIFO_ID_FIELD_POSITION ) | ( uint32_t )ucPayload;

            vSMBusHWWriteTgtDescFifo( pxSMBusProfile, ulValue );
        }
    }
    return ucReturnCode;
}

/******************************************************************************
*
* @brief    Given the descriptor, look up the descriptor ID and write it into the
*           IP's Controller descriptor FIFO. Depending on the descriptor type a
*           payload byte may be written along with the descriptor.
*           The ucNoStatusCheck can be used to check if the FIFO is full before
*           writing the descriptor or ignore the check
*
*****************************************************************************/
static uint8_t prvucSMBusControllerDescriptorApply( SMBUS_PROFILE_TYPE* pxSMBusProfile,
                                                    SMBus_HW_Descriptor_Type xDescriptor, uint8_t ucPayload,
                                                    uint8_t ucNoStatusCheck )
{
    uint8_t  ucReturnCode   = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    uint8_t  ucDescriptorID = 0;
    uint32_t ulValue        = 0;

    if( ( NULL != pxSMBusProfile ) &&
        ( DESC_MAX > xDescriptor ) )
    {
        ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS;

        if( SMBUS_FALSE == ucNoStatusCheck )
        {
            if( SMBUS_DESC_FIFO_IS_FULL == ulSMBusHWReadCtrlDescStatusFull( pxSMBusProfile ) )
            {
                ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
            }
        }

        if( SMBUS_HW_DESCRIPTOR_WRITE_SUCCESS == ucReturnCode )
        {
            ucDescriptorID = prvucSMBusGenericDescriptorIDLookup( xDescriptor );
            ulValue = ( ( uint32_t )ucDescriptorID << SMBUS_CTLR_DESC_FIFO_ID_FIELD_POSITION ) | ( uint32_t )ucPayload;

            vSMBusHWWriteCtrlDescFifo( pxSMBusProfile, ulValue );
        }
    }
    return ucReturnCode;
}

/*******************************************************************************
*
* @brief    Writes a data byte along with a Target Read - Read Descriptor ID to transmit the data byte
*
*******************************************************************************/
uint8_t ucSMBusTargetReadDescriptorRead( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucData )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;

    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusTargetDescriptorApply( pxSMBusProfile, DESC_TARGET_READ, ucData, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Target Read - PEC Descriptor ID to the Target Descriptor FIFO
* To iform the IP to transmit the PEC byte it has calculated on the data
*
*****************************************************************/
uint8_t ucSMBusTargetReadDescriptorPECRead( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusTargetDescriptorApply( pxSMBusProfile, DESC_TARGET_READ_PEC, 0, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Target Write - ACK Descriptor ID to the Target Descriptor FIFO
*           To inform the IP to transmit an ACK
*
*******************************************************************************/
uint8_t ucSMBusTargetWriteDescriptorACK( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucNoStatusCheck )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusTargetDescriptorApply( pxSMBusProfile, DESC_TARGET_WRITE_ACK, 0, ucNoStatusCheck );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Target Write - NACK Descriptor ID to the Target Descriptor FIFO
*           To inform the IP to transmit a NACK
*
*******************************************************************************/
uint8_t ucSMBusTargetWriteDescriptorNACK( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusTargetDescriptorApply( pxSMBusProfile, DESC_TARGET_WRITE_NACK, 0, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Target Write - PEC Descriptor ID to the Target Descriptor FIFO
*           To inform the IP to interpret the previous byte as a PEC
*
*******************************************************************************/
uint8_t ucSMBusTargetWriteDescriptorPEC( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusTargetDescriptorApply( pxSMBusProfile, DESC_TARGET_WRITE_PEC, 0, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Write - START Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a START condition
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorStartWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        /* Destination is a 7-bit address occupying bits 7:1, bit 0 is the r/W bit
           This is a write so set the R/W bit to Zero */
        ucDestination = ( ucDestination << 1 );
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_WRITE_START, ucDestination, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Write - QUICK Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a START condition followed by a STOP
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorQuickWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucDestination = ( ucDestination << 1 );
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_WRITE_QUICK, ucDestination, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Write - BYTE Descriptor ID to the Controller Descriptor FIFO
* along with a data byte
*           To inform the IP to transmit the data byte
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorByte( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucData, uint8_t ucNoStatusCheck )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_WRITE_BYTE, ucData, ucNoStatusCheck );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Write - STOP Descriptor ID to the Controller Descriptor FIFO
* along with a data byte
*           To inform the IP to transmit the data byte followed by a STOP condition
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorStopWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucData )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_WRITE_STOP, ucData, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Write - PEC Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit the PEC verify an ACK and then transmit a STOP
*
*******************************************************************************/
uint8_t ucSMBusControllerWriteDescriptorPECWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_WRITE_PEC, 0, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Read - START Descriptor ID to the Controller Descriptor FIFO
* along with Target address
*           To inform the IP to start a new READ transaction
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorStart( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        /* Destination is a 7-bit address occupying bits 7:1, bit 0 is the r/W bit
           This is a read so set the R/W bit to One */
        ucDestination = ( ucDestination << 1 ) | 0x01;
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_READ_START, ucDestination, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Read - QUICK Descriptor ID to the Controller Descriptor FIFO
* along with Target address
*           To inform the IP to start a new READ transaction followed by a STOP
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorQuickRead( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucDestination )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucDestination = ( ucDestination << 1 ) | 0x01;
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_READ_QUICK, ucDestination, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Read - BYTE Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit an ACK for the previous byte received
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorByte( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucNoStatusCheck )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_READ_BYTE, 0, ucNoStatusCheck );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Read - STOP Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a NACK followed by a STOP condition
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorStop( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_READ_STOP, 0, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/*******************************************************************************
*
* @brief    Writes a Controller Read - PEC Descriptor ID to the Controller Descriptor FIFO
*           To inform the IP to transmit a NACK followed by a STOP and use the last received byte
*           to perform a PEC check
*
*******************************************************************************/
uint8_t ucSMBusControllerReadDescriptorPEC( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucReturnCode = SMBUS_HW_DESCRIPTOR_WRITE_FAIL;
    if( NULL != pxSMBusProfile )
    {
        ucReturnCode = prvucSMBusControllerDescriptorApply( pxSMBusProfile, DESC_CONTROLLER_READ_PEC, 0, SMBUS_FALSE );
    }
    return ( ucReturnCode );
}

/* Descriptor helper functions*/

/*******************************************************************************
*
* @brief    Converts an Descriptor enum value to a text string for logging
*
*******************************************************************************/
char* pcDescriptorToString( SMBus_HW_Descriptor_Type xDescriptor )
{
    char* pResult = NULL;

    switch( xDescriptor )
    {
    case DESC_TARGET_READ:
        pResult = pDescriptorTargetRead;
        break;
    
    case DESC_TARGET_READ_PEC:
        pResult = pDescriptorTargetPEC;
        break;
    
    case DESC_TARGET_WRITE_ACK:
        pResult = pDescriptorTargetWriteACK;
        break;
    
    case DESC_TARGET_WRITE_NACK:
        pResult = pDescriptorTargetWriteNACK;
        break;
    
    case DESC_TARGET_WRITE_PEC:
        pResult = pDescriptorTargetWritePEC;
        break;
    
    case DESC_CONTROLLER_READ_START:
        pResult = pDescriptorControllerReadStart;
        break;
    
    case DESC_CONTROLLER_READ_QUICK:
        pResult = pDescriptorControllerReadQuick;
        break;
    
    case DESC_CONTROLLER_READ_BYTE:
        pResult = pDescriptorControllerReadByte;
        break;
    
    case DESC_CONTROLLER_READ_STOP:
        pResult = pDescriptorControllerReadStop;
        break;
    
    case DESC_CONTROLLER_READ_PEC:
        pResult = pDescriptorControllerReadPEC;
        break;
    
    case DESC_CONTROLLER_WRITE_START:
        pResult = pDescriptorControllerWriteStart;
        break;
    
    case DESC_CONTROLLER_WRITE_QUICK:
        pResult = pDescriptorControllerWriteQuick;
        break;
    
    case DESC_CONTROLLER_WRITE_BYTE:
        pResult = pDescriptorControllerWriteByte;
        break;
    
    case DESC_CONTROLLER_WRITE_STOP:
        pResult = pDescriptorControllerWriteStop;
        break;
    
    case DESC_CONTROLLER_WRITE_PEC:
        pResult = pDescriptorControllerWritePEC;
        break;
    
    default:
        pResult = pDescriptorUnknown;
        break;
    }

    return pResult;
}

/******************************************************************************
*
* @brief    Return the value read form the register offset location
*
*****************************************************************************/
static uint32_t prvulSMBusHardwareRead( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulRegisterOffset )
{
    uint32_t           ulReadValue = 0;
    volatile uintptr_t xAddress    = 0;
    
    if( NULL != pxSMBusProfile )
    {
        xAddress = ( uintptr_t )( ( ( SMBUS_BASE_ADDRESS_TYPE )pxSMBusProfile->pvBaseAddress ) + ulRegisterOffset / 4 );
        ulReadValue = prvulSMBusIn32( ( void* )xAddress );
        
        vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG, 
                        SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_HW_READ, ( uint32_t )ulRegisterOffset, ( uint32_t )ulReadValue );
    }
 
    return ( ulReadValue );
}

/******************************************************************************
*
* @brief    Write the supplied value to the register offset location
*
*****************************************************************************/
static void prvvSMBusHardwareWrite( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulRegisterOffset, uint32_t ulValue )
{
    volatile uintptr_t xAddress = 0;
    
    if( NULL != pxSMBusProfile )
    {
        xAddress = ( uintptr_t )( ( ( SMBUS_BASE_ADDRESS_TYPE )pxSMBusProfile->pvBaseAddress ) + ulRegisterOffset / 4 );
        
        vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_DEBUG,
                        SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_HW_WRITE, ( uint32_t )ulRegisterOffset, ( uint32_t )ulValue );
        prvvSMBusOut32( ( void* )xAddress, ulValue );
    }
}

/******************************************************************************
*
* @brief    Modify the masked part of the register offset location with the value supplied
*
*****************************************************************************/
static void prvvSMBusHardwareWriteWithMask( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulRegisterOffset,
                                    uint32_t ulMask, uint32_t ulValue )
{
    
    uint32_t ulRegisterulValue = 0;
    uint32_t ulMaskedulValue   = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulRegisterulValue = prvulSMBusHardwareRead( pxSMBusProfile, ulRegisterOffset );
        ulRegisterulValue = ulRegisterulValue & ~ulMask;
        ulMaskedulValue = ulValue & ulMask;

        ulRegisterulValue = ulRegisterulValue | ulMaskedulValue;

        prvvSMBusHardwareWrite( pxSMBusProfile, ulRegisterOffset, ulRegisterulValue );
    }
}

/* SMBUS_REG_IP_VERSION */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_VERSION
*
*******************************************************************************/
uint32_t ulSMBusHWReadIPVersion( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue = ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IP_VERSION ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IP_REVISION */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_REVISION
*
*******************************************************************************/
uint32_t ulSMBusHWReadIPRevision( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IP_REVISION ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IP_MAGIC_NUM */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IP_MAGIC_NUM
*
*******************************************************************************/
uint32_t ulSMBusHWReadIPMagicNum( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IP_MAGIC_NUM ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_BUILD_CONFIG_0 */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_BUILD_CONFIG_0
*
*******************************************************************************/
uint32_t ulSMBusHWReadBuildConfig0( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IP_BUILD_CONFIG_0 ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_BUILD_CONFIG_1 */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_BUILD_CONFIG_1
*
*******************************************************************************/
uint32_t ulSMBusHWReadBuildConfig1( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IP_BUILD_CONFIG_1 ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IRQ_GIE */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_GIE
*
*******************************************************************************/
uint32_t ulSMBusHWReadIRQGIEEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IRQ_GIE ) & SMBUS_IRQ_GIE_ENABLE_MASK );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IRQ_IER */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_IER
*
*******************************************************************************/
uint32_t ulSMBusHWReadIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IRQ_IER ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IRQ_ISR */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_ISR
*
*******************************************************************************/
uint32_t ulSMBusHWReadIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_IRQ_ISR ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IRQ_ERR_IER */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_ERR_IER
*
*******************************************************************************/
uint32_t ulSMBusHWReadErrIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_ERR_IRQ_IER ) );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_IRQ_ERR_ISR */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_IRQ_ERR_ISR
*
*******************************************************************************/
uint32_t ulSMBusHWReadErrIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_ERR_IRQ_ISR ) );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_STATUS */
/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_SMBDAT_LOW_TIMEOUT bitfield from hardware
*           register SMBUS_REG_PHY_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatus( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_STATUS ) );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_SMBCLK_LOW_TIMEOUT bitfield from hardware
*           register SMBUS_REG_PHY_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatusSMBDATLowTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_STATUS ) 
            & SMBUS_PHY_STATUS_SMBDAT_LOW_TIMEOUT_MASK ) >> SMBUS_PHY_STATUS_SMBDAT_LOW_TIMEOUT_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_SMBCLK_LOW_TIMEOUT bitfield from hardware
*           register SMBUS_REG_PHY_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatusSMBClkLowTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_STATUS ) 
            & SMBUS_PHY_STATUS_SMBCLK_LOW_TIMEOUT_MASK ) >> SMBUS_PHY_STATUS_SMBCLK_LOW_TIMEOUT_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_STATUS_BUS_IDLE bitfield from hardware register
*           SMBUS_REG_PHY_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYStatusBusIdle( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_STATUS ) & SMBUS_PHY_STATUS_BUS_IDLE_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_FILTER_CONTROL */
/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_FILTER_CONTROL_ENABLE bitfield from hardware
*           register SMBUS_REG_PHY_FILTER_CONTROL
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYFilterControl( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_FILTER_CONTROL ) );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_FILTER_CONTROL_ENABLE bitfield from hardware
*           register SMBUS_REG_PHY_FILTER_CONTROL
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYFilterControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_FILTER_CONTROL ) 
            & SMBUS_PHY_FILTER_CONTROL_ENABLE_MASK ) >> SMBUS_PHY_FILTER_CONTROL_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_FILTER_CONTROL_DURATION bitfield from hardware
*           register SMBUS_REG_PHY_FILTER_CONTROL
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYFilterControlDuration( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_FILTER_CONTROL ) 
            & SMBUS_PHY_FILTER_CONTROL_DURATION_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_BUS_FREE_TIME */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_BUS_FREE_TIME
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYBusFreetime( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_BUS_FREE_TIME ) );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_IDLE_THRESHOLD */
/*******************************************************************************
*
* @brief    Reads the IDLE_THRESHOLD bitfield from hardware register
*           SMBUS_REG_PHY_IDLE_THRESHOLD
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYIdleThresholdIdleThreshold( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_IDLE_THRESHOLD ) 
            & SMBUS_PHY_IDLE_THRESHOLD_IDLE_THRESHOLD_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TIMEOUT_PRESCALER */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_TIMEOUT_PRESCALER
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_PRESCALER ) );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TIMEOUT_MIN */
/*******************************************************************************
*
* @brief    Reads the MIN_TIMEOUT_ENABLE bitfield from hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MIN ) );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the MIN_TIMEOUT_MIN bitfield from hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMinTimeoutEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MIN ) 
            & SMBUS_PHY_TIMEOUT_MIN_TIMEOUT_ENABLE_MASK ) >> SMBUS_PHY_TIMEOUT_MIN_TIMEOUT_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the SMBUS_PHY_FILTER_CONTROL_DURATION bitfield from hardware
*           register SMBUS_REG_PHY_FILTER_CONTROL
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMinTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MIN ) 
            & SMBUS_PHY_TIMEOUT_MIN_TIMEOUT_MIN_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TIMEOUT_MAX */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_PHY_TIMEOUT_MAX
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTimeoutMax( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MAX ) );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_RESET_CONTROL */
/*******************************************************************************
*
* @brief    Reads the SMBCLK_FORCE_LOW bitfield from hardware register
*           SMBUS_REG_PHY_RESET_CONTROL
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYResetControlSMBClkForce( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_RESET_CONTROL ) 
            & SMBUS_PHY_RESET_CONTROL_SMBCLK_FORCE_LOW_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_DATA_SETUP */
/*******************************************************************************
*
* @brief    Reads the TGT_DATA_SETUP bitfield from hardware register
*           SMBUS_REG_PHY_TGT_DATA_SETUP
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtDataSetupTgtDataSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TGT_DATA_SETUP ) 
            & SMBUS_PHY_TGT_DATA_SETUP_TGT_DATA_SETUP_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_TEXT_PRESCALER */
/*******************************************************************************
*
* @brief    Reads the TGT_TEXT_PRESCALER bitfield from hardware register
*           SMBUS_REG_PHY_TGT_TEXT_PRESCALER
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TGT_TEXT_PRESCALER ) 
            & SMBUS_PHY_TGT_TEXT_PRESCALER_TGT_TEXT_PRESCALER_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_TEXT_TIMEOUT */
/*******************************************************************************
*
* @brief    Reads the TGT_TEXT_TIMEOUT bitfield from hardware register
*           SMBUS_REG_PHY_TGT_TEXT_TIMEOUT
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TGT_TEXT_TIMEOUT ) 
            & SMBUS_PHY_TGT_TEXT_TIMEOUT_TGT_TEXT_TIMEOUT_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_TEXT_MAX */
/*******************************************************************************
*
* @brief    Reads the TGT_TEXT_MAX bitfield from hardware register
*           SMBUS_REG_PHY_TGT_TEXT_MAX
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TGT_TEXT_MAX ) 
            & SMBUS_PHY_TGT_TEXT_MAX_TGT_TEXT_MAX_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_DBG_STATE */
/*******************************************************************************
*
* @brief    Reads the DBG_STATE bitfield from hardware register
*           SMBUS_REG_PHY_TGT_DBG_STATE
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtDbgState( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TGT_DBG_STATE ) 
            & SMBUS_PHY_TGT_DBG_STATE_DBG_STATE_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_DATA_HOLD */
/*******************************************************************************
*
* @brief    Reads the DATA_HOLD bitfield from hardware register
*           SMBUS_REG_PHY_TGT_DATA_HOLD
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYTgtDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_TGT_DATA_HOLD ) 
            & SMBUS_PHY_TGT_DATA_HOLD_DATA_HOLD_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_STATUS */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_TGT_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatus( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_STATUS ) );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the TGT_STATUS_ACTIVE bitfield from hardware register
*           SMBUS_REG_TGT_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatusActive( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_STATUS ) 
            & SMBUS_TGT_STATUS_ACTIVE_MASK ) >> SMBUS_TGT_STATUS_ACTIVE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the TGT_STATUS_ADDRESS bitfield from hardware register
*           SMBUS_REG_TGT_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatusAddress( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_STATUS ) 
            & SMBUS_TGT_STATUS_ADDRESS_MASK ) >> SMBUS_TGT_STATUS_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the TGT_STATUS_RW bitfield from hardware register
*           SMBUS_REG_TGT_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtStatusRW( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_STATUS ) 
            & SMBUS_TGT_STATUS_RW_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_DESC_STATUS */
/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_DESC_STATUS ) 
            & SMBUS_TGT_DESC_STATUS_FILL_LEVEL_MASK ) >> SMBUS_TGT_DESC_STATUS_FILL_LEVEL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_DESC_STATUS ) 
            & SMBUS_TGT_DESC_STATUS_FULL_MASK ) >> SMBUS_TGT_DESC_STATUS_FULL_FIELD_POSITION );
    }
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_DESC_STATUS ) 
            & SMBUS_TGT_DESC_STATUS_ALMOST_FULL_MASK ) >> SMBUS_TGT_DESC_STATUS_ALMOST_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_DESC_STATUS ) 
            & SMBUS_TGT_DESC_STATUS_ALMOST_EMPTY_MASK ) >> SMBUS_TGT_DESC_STATUS_ALMOST_EMPTY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register
*           SMBUS_REG_TGT_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDescStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_DESC_STATUS ) 
            & SMBUS_TGT_DESC_STATUS_EMPTY_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_RX_FIFO */
/*******************************************************************************
*
* @brief    Reads the PAYLOAD bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoPayload( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO ) 
            & SMBUS_TGT_RX_FIFO_PAYLOAD_MASK ) >> SMBUS_TGT_RX_FIFO_PAYLOAD_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_RX_FIFO_STATUS */
/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_STATUS ) 
            & SMBUS_TGT_RX_FIFO_STATUS_FILL_LEVEL_MASK ) >> SMBUS_TGT_RX_FIFO_STATUS_FILL_LEVEL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the RESET_BUSY bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusResetBusY( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_STATUS ) 
            & SMBUS_TGT_RX_FIFO_STATUS_RESET_BUSY_MASK ) >> SMBUS_TGT_RX_FIFO_STATUS_RESET_BUSY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_STATUS ) 
            & SMBUS_TGT_RX_FIFO_STATUS_FULL_MASK ) >> SMBUS_TGT_RX_FIFO_STATUS_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_STATUS ) 
            & SMBUS_TGT_RX_FIFO_STATUS_ALMOST_FULL_MASK ) >> SMBUS_TGT_RX_FIFO_STATUS_ALMOST_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_STATUS ) 
            & SMBUS_TGT_RX_FIFO_STATUS_ALMOST_EMPTY_MASK ) >> SMBUS_TGT_RX_FIFO_STATUS_ALMOST_EMPTY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_STATUS ) 
            & SMBUS_TGT_RX_FIFO_STATUS_EMPTY_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD */
/*******************************************************************************
*
* @brief    Reads the FILL_THRESHOLD bitfield from hardware register
*           SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtRxFifoFillThresholdFillThresh( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD ) 
            & SMBUS_TGT_RX_FIFO_FILL_THRESHOLD_FILL_THRESHOLD_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_DBG */
/*******************************************************************************
*
* @brief    Reads the hardware register SMBUS_REG_TGT_DBG
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtDbg( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_DBG ) 
            & SMBUS_TGT_DBG_DBG_STATE_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_CONTROL_0 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_0
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl0Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_0 ) 
            & SMBUS_TGT_CONTROL_0_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_0_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_0
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl0Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_0 ) 
            & SMBUS_TGT_CONTROL_0_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_0_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_1 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_1
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl1Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_1 ) 
            & SMBUS_TGT_CONTROL_1_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_1_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_1
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl1Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_1 ) 
            & SMBUS_TGT_CONTROL_1_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_1_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_2 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_2
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl2Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_2 ) 
            & SMBUS_TGT_CONTROL_2_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_2_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_2
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl2Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_2 ) 
            & SMBUS_TGT_CONTROL_2_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_2_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_3 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_3
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl3Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_3 ) 
            & SMBUS_TGT_CONTROL_3_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_3_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_3
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl3Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_3 ) 
            & SMBUS_TGT_CONTROL_3_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_3_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_4 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_4
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl4Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_4 ) 
            & SMBUS_TGT_CONTROL_4_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_4_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_4
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl4Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_4 ) 
            & SMBUS_TGT_CONTROL_4_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_4_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_5 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_5
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl5Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_5 ) 
            & SMBUS_TGT_CONTROL_5_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_5_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_5
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl5Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_5 ) 
            & SMBUS_TGT_CONTROL_5_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_5_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_6 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_6
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl6Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_6 ) 
            & SMBUS_TGT_CONTROL_6_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_6_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_6
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl6Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_6 ) 
            & SMBUS_TGT_CONTROL_6_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_6_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/* SMBUS_REG_TGT_CONTROL_7 */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_TGT_CONTROL_7
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl7Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_7 ) 
            & SMBUS_TGT_CONTROL_7_ENABLE_MASK ) >> SMBUS_TGT_CONTROL_7_ENABLE_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ADDRESS bitfield from hardware register SMBUS_REG_TGT_CONTROL_7
*
*******************************************************************************/
uint32_t ulSMBusHWReadTgtControl7Address( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_7 ) 
            & SMBUS_TGT_CONTROL_7_ADDRESS_MASK ) >> SMBUS_TGT_CONTROL_7_ADDRESS_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_DATA_HOLD */
/*******************************************************************************
*
* @brief    Reads the DATA_HOLD bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_DATA_HOLD
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_DATA_HOLD ) 
            & SMBUS_PHY_CTLR_DATA_HOLD_CTLR_DATA_HOLD_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_START_HOLD */
/*******************************************************************************
*
* @brief    Reads the START_HOLD bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_START_HOLD
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlStartHold( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_START_HOLD ) 
            & SMBUS_PHY_CTLR_START_HOLD_CTLR_START_HOLD_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_START_SETUP */
/*******************************************************************************
*
* @brief    Reads the START_SETUP bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_START_SETUP
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlStartSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_START_SETUP ) 
            & SMBUS_PHY_CTLR_START_SETUP_CTLR_START_SETUP_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_STOP_SETUP */
/*******************************************************************************
*
* @brief    Reads the STOP_SETUP bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_STOP_SETUP
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlStopSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_STOP_SETUP ) 
            & SMBUS_PHY_CTLR_STOP_SETUP_CTLR_STOP_SETUP_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CLK_TLOW */
/*******************************************************************************
*
* @brief    Reads the CLK_TLOW bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CLK_TLOW
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlClkTLow( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CLK_TLOW ) 
            & SMBUS_PHY_CTLR_CLK_TLOW_CTLR_CLK_TLOW_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CLK_THIGH */
/*******************************************************************************
*
* @brief    Reads the CLK_THIGH bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CLK_THIGH
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlClkTHigh( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CLK_THIGH ) 
            & SMBUS_PHY_CTLR_CLK_THIGH_CTLR_CLK_THIGH_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_TEXT_PRESCALER */
/*******************************************************************************
*
* @brief    Reads the TEXT_PRESCALER bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_TEXT_PRESCALER
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_TEXT_PRESCALER ) 
            & SMBUS_PHY_CTLR_TEXT_PRESCALER_CTLR_TEXT_PRESCALER_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT */
/*******************************************************************************
*
* @brief    Reads the TEXT_TIMEOUT bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT ) 
            & SMBUS_PHY_CTLR_TEXT_TIMEOUT_CTLR_TEXT_TIMEOUT_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_TEXT_MAX */
/*******************************************************************************
*
* @brief    Reads the TEXT_MAX bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_TEXT_MAX
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_TEXT_MAX ) 
            & SMBUS_PHY_CTLR_TEXT_MAX_CTLR_TEXT_MAX_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CEXT_PRESCALER */
/*******************************************************************************
*
* @brief    Reads the CEXT_PRESCALER bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CEXT_PRESCALER
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlCextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CEXT_PRESCALER ) 
            & SMBUS_PHY_CTLR_CEXT_PRESCALER_CTLR_CEXT_PRESCALER_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT */
/*******************************************************************************
*
* @brief    Reads the CEXT_TIMEOUT bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlCextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT ) 
            & SMBUS_PHY_CTLR_CEXT_TIMEOUT_CTLR_CEXT_TIMEOUT_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CEXT_MAX */
/*******************************************************************************
*
* @brief    Reads the CEXT_MAX bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_CEXT_MAX
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlCextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CEXT_MAX ) 
            & SMBUS_PHY_CTLR_CEXT_MAX_CTLR_CEXT_MAX_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_DBG_STATE */
/*******************************************************************************
*
* @brief    Reads the DBG_STATE bitfield from hardware register
*           SMBUS_REG_PHY_CTLR_DBG_STATE
*
*******************************************************************************/
uint32_t ulSMBusHWReadPHYCtrlDbgState( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_PHY_CTLR_DBG_STATE ) 
            & SMBUS_PHY_CTLR_DBG_STATE_DBG_STATE_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_STATUS */
/*******************************************************************************
*
* @brief    Reads the ENABLE bitfield from hardware register SMBUS_REG_CTLR_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlStatusEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_STATUS ) 
            & SMBUS_CTLR_STATUS_ENABLE_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_DESC_STATUS */
/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DESC_STATUS ) 
            & SMBUS_CTLR_DESC_STATUS_FILL_LEVEL_MASK ) >> SMBUS_CTLR_DESC_STATUS_FILL_LEVEL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the RESET_BUSY bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusResetBusy( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DESC_STATUS ) 
            & SMBUS_CTLR_DESC_STATUS_RESET_BUSY_MASK ) >> SMBUS_CTLR_DESC_STATUS_RESET_BUSY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DESC_STATUS ) 
            & SMBUS_CTLR_DESC_STATUS_FULL_MASK ) >> SMBUS_CTLR_DESC_STATUS_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DESC_STATUS ) 
            & SMBUS_CTLR_DESC_STATUS_ALMOST_FULL_MASK ) >> SMBUS_CTLR_DESC_STATUS_ALMOST_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DESC_STATUS ) 
            & SMBUS_CTLR_DESC_STATUS_ALMOST_EMPTY_MASK ) >> SMBUS_CTLR_DESC_STATUS_ALMOST_EMPTY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_DESC_STATUS
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDescStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DESC_STATUS ) 
            & SMBUS_CTLR_DESC_STATUS_EMPTY_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_RX_FIFO */
/*******************************************************************************
*
* @brief    Reads the PAYLOAD bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoPayload( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO ) 
            & SMBUS_CTLR_RX_FIFO_PAYLOAD_MASK ) >> SMBUS_CTLR_RX_FIFO_PAYLOAD_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_RX_FIFO_STATUS */
/*******************************************************************************
*
* @brief    Reads the FILL_LEVEL bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusFillLevel( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_STATUS ) 
            & SMBUS_CTLR_RX_FIFO_STATUS_FILL_LEVEL_MASK ) >> SMBUS_CTLR_RX_FIFO_STATUS_FILL_LEVEL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the RESET_BUSY bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusResetBusy( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_STATUS ) 
            & SMBUS_CTLR_RX_FIFO_STATUS_RESET_BUSY_MASK ) >> SMBUS_CTLR_RX_FIFO_STATUS_RESET_BUSY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the FULL bitfield from hardware register SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_STATUS ) 
            & SMBUS_CTLR_RX_FIFO_STATUS_FULL_MASK ) >> SMBUS_CTLR_RX_FIFO_STATUS_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_FULL bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusAlmostFull( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_STATUS ) 
            & SMBUS_CTLR_RX_FIFO_STATUS_ALMOST_FULL_MASK ) >> SMBUS_CTLR_RX_FIFO_STATUS_ALMOST_FULL_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the ALMOST_EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusAlmostEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( ( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_STATUS ) 
            & SMBUS_CTLR_RX_FIFO_STATUS_ALMOST_EMPTY_MASK ) >> SMBUS_CTLR_RX_FIFO_STATUS_ALMOST_EMPTY_FIELD_POSITION );
    }
    
    return( ulReadValue );
}

/*******************************************************************************
*
* @brief    Reads the EMPTY bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoStatusEmpty( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_STATUS ) 
            & SMBUS_CTLR_RX_FIFO_STATUS_EMPTY_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD */
/*******************************************************************************
*
* @brief    Reads the THRESHOLD bitfield from hardware register
*           SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlRxFifoFillThresholdFillThresh( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD ) 
            & SMBUS_CTLR_RX_FIFO_FILL_THRESHOLD_FILL_THRESHOLD_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_DBG_STATE */
/*******************************************************************************
*
* @brief    Reads the DBG_STATE bitfield from hardware register
*           SMBUS_REG_CTLR_DBG
*
*******************************************************************************/
uint32_t ulSMBusHWReadCtrlDbgState( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint32_t ulReadValue = 0;
    
    if( NULL != pxSMBusProfile )
    {
        ulReadValue =( prvulSMBusHardwareRead( pxSMBusProfile, SMBUS_REG_CTLR_DBG ) 
            & SMBUS_CTLR_DBG_DBG_STATE_MASK );
    }
    
    return( ulReadValue );
}

/******************************************************************************************************************/

/******************************************************************************************************************/
/* Write Functions */
/******************************************************************************************************************/

/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_IRQ_GIE
*
*******************************************************************************/
void vSMBusHWWriteIRQGIEEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_IRQ_GIE, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_IRQ_IER */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_IRQ_IER
*
*******************************************************************************/
void vSMBusHWWriteIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {   
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_IRQ_IER, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_IRQ_ISR */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_IRQ_ISR
*
*******************************************************************************/
void vSMBusHWWriteIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_IRQ_ISR, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_ERR_IRQ_IER */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_ERR_IRQ_IER
*
*******************************************************************************/
void vSMBusHWWriteERRIRQIER( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_ERR_IRQ_IER, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_ERR_IRQ_ISR */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_ERR_IRQ_ISR
*
*******************************************************************************/
void vSMBusHWWriteERRIRQISR( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_ERR_IRQ_ISR, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_FILTER_CONTROL */
/*******************************************************************************
*
* @brief    Writes ulValue to ENABLE bitfield of hardware register
*           SMBUS_REG_PHY_FILTER_CONTROL
*
*******************************************************************************/
void vSMBusHWWritePHYFilterControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_PHY_FILTER_CONTROL_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_FILTER_CONTROL, SMBUS_PHY_FILTER_CONTROL_ENABLE_MASK, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to DURATION bitfield of hardware register
*           SMBUS_REG_PHY_FILTER_CONTROL
*
*******************************************************************************/
void vSMBusHWWritePHYFilterControlDuration( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {   
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_FILTER_CONTROL, SMBUS_PHY_FILTER_CONTROL_DURATION_MASK, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_BUS_FREE_TIME */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_BUS_FREE_TIME
*
*******************************************************************************/
void vSMBusHWWritePHYBusFreetime( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {   
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_BUS_FREE_TIME, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_IDLE_THRESHOLD */
/*******************************************************************************
*
* @brief    Writes ulValue to IDLE_THRESHOLD bitfield of hardware register
*           SMBUS_REG_PHY_IDLE_THRESHOLD
*
*******************************************************************************/
void vSMBusHWWritePHYIdleThresholdIdleThreshold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {   
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_IDLE_THRESHOLD, 
                            SMBUS_PHY_IDLE_THRESHOLD_IDLE_THRESHOLD_MASK, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TIMEOUT_PRESCALER */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TIMEOUT_PRESCALER
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_PRESCALER, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TIMEOUT_MIN */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TIMEOUT_MIN
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MIN, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to ENABLE bitfield of hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMinTimeoutEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_PHY_TIMEOUT_MIN_TIMEOUT_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MIN, SMBUS_PHY_TIMEOUT_MIN_TIMEOUT_ENABLE_MASK, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to TIMEOUT_MIN bitfield of hardware register
*           SMBUS_REG_PHY_TIMEOUT_MIN
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMinTimeoutMin( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MIN, SMBUS_PHY_TIMEOUT_MIN_TIMEOUT_MIN_MASK, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TIMEOUT_MAX */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TIMEOUT_MAX
*
*******************************************************************************/
void vSMBusHWWritePHYTimeoutMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TIMEOUT_MAX, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_RESET_CONTROL */
/*******************************************************************************
*
* @brief    Writes ulValue to SMBCLK_FORCE_TIMEOUT bitfield of hardware register
*           SMBUS_REG_PHY_RESET_CONTROL
*
*******************************************************************************/
void vSMBusHWWritePHYResetControlSMBClkForceTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_PHY_RESET_CONTROL_SMBCLK_FORCE_TIMEOUT_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_RESET_CONTROL, 
                                SMBUS_PHY_RESET_CONTROL_SMBCLK_FORCE_TIMEOUT_MASK, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to SMBCLK_FORCE_LOW bitfield of hardware register
*           SMBUS_REG_PHY_RESET_CONTROL
*
*******************************************************************************/
void vSMBusHWWritePHYResetControlSMBClkForceLow( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_RESET_CONTROL, 
                                SMBUS_PHY_RESET_CONTROL_SMBCLK_FORCE_LOW_MASK, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_DATA_SETUP */
/*******************************************************************************
*
* @brief    Writes ulValue to TGT_DATA_SETUP bitfield of hardware register
*           SMBUS_REG_PHY_TGT_DATA_SETUP
*
*******************************************************************************/
void vSMBusHWWritePHYTgtDataSetupTgtDataSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_PHY_TGT_DATA_SETUP, 
                                SMBUS_PHY_TGT_DATA_SETUP_TGT_DATA_SETUP_MASK, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_TEXT_PRESCALER */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_TEXT_PRESCALER
*
*******************************************************************************/
void vSMBusHWWritePHYTgtTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TGT_TEXT_PRESCALER, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_TEXT_TIMEOUT */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_TEXT_TIMEOUT
*
*******************************************************************************/
void vSMBusHWWritePHYTgtTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TGT_TEXT_TIMEOUT, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_TEXT_MAX */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_TEXT_MAX
*
*******************************************************************************/
void vSMBusHWWritePHYTgtTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TGT_TEXT_MAX, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_TGT_DATA_HOLD */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_TGT_DATA_HOLD
*
*******************************************************************************/
void vSMBusHWWritePHYTgtDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_TGT_DATA_HOLD, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_DESC_FIFO */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_TGT_DESC_FIFO
*
*******************************************************************************/
void vSMBusHWWriteTgtDescFifo( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_TGT_DESC_FIFO, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to FIFO_ID bitfield of hardware register
*           SMBUS_REG_TGT_DESC_FIFO
*
*******************************************************************************/
void vSMBusHWWriteTgtDescFifoId( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_DESC_FIFO_ID_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_DESC_FIFO, SMBUS_TGT_DESC_FIFO_ID_MASK, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to PAYLOAD bitfield of hardware register
*           SMBUS_REG_TGT_DESC_FIFO
*
*******************************************************************************/
void vSMBusHWWriteTgtDescFifoPayload( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_DESC_FIFO, SMBUS_TGT_DESC_FIFO_PAYLOAD_MASK, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_RX_FIFO */
/*******************************************************************************
*
* @brief    Writes ulValue to RESET bitfield of hardware register
*           SMBUS_REG_TGT_RX_FIFO
*
*******************************************************************************/
void vSMBusHWWriteTgtRxFifoReset( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_RX_FIFO_RESET_FIELD_POSITION;
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD */
/*******************************************************************************
*
* @brief    Writes ulValue to THRESHOLD bitfield of hardware register
*           SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD
*
*******************************************************************************/
void vSMBusHWWriteRxFifoFillThresholdFillThresh( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_TGT_RX_FIFO_FILL_THRESHOLD, 
                        ( SMBUS_TGT_RX_FIFO_FILL_THRESHOLD_FILL_THRESHOLD_MASK & ulValue ) );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_TGT_CONTROL_0 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_0
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl0Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_0_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_0, SMBUS_TGT_CONTROL_0_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_0
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl0Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_0_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_0, SMBUS_TGT_CONTROL_0_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_1 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_1
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl1Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_1_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_1, SMBUS_TGT_CONTROL_1_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_1
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl1Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_1_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_1, SMBUS_TGT_CONTROL_1_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_2 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_2
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl2Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_2_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_2, SMBUS_TGT_CONTROL_2_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_2
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl2Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_2_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_2, SMBUS_TGT_CONTROL_2_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_3 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_3
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl3Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_3_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_3, SMBUS_TGT_CONTROL_3_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_3
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl3Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_3_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_3, SMBUS_TGT_CONTROL_3_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_4 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_4
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl4Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_4_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_4, SMBUS_TGT_CONTROL_4_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_4
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl4Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_4_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_4, SMBUS_TGT_CONTROL_4_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_5 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_5
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl5Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_5_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_5, SMBUS_TGT_CONTROL_5_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_5
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl5Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
    ulValue = ulValue << SMBUS_TGT_CONTROL_5_ADDRESS_FIELD_POSITION;
    prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_5, SMBUS_TGT_CONTROL_5_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_6 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_6
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl6Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_6_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_6, SMBUS_TGT_CONTROL_6_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_6
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl6Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_6_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_6, SMBUS_TGT_CONTROL_6_ADDRESS_MASK, ulValue );
    }
}

/* SMBUS_REG_TGT_CONTROL_7 */
/******************************************************************************
*
* @brief    Write ulValue to the ENABLE bitfield of the SMBUS_REG_TGT_CONTROL_7
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl7Enable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_7_ENABLE_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_7, SMBUS_TGT_CONTROL_7_ENABLE_MASK, ulValue );
    }
}

/******************************************************************************
*
* @brief    Write ulValue to the ADDRESS bitfield of the SMBUS_REG_TGT_CONTROL_7
*           register
*
*****************************************************************************/
static void prvvSMBusHWWriteTgtControl7Address( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_TGT_CONTROL_7_ADDRESS_FIELD_POSITION;
        prvvSMBusHardwareWriteWithMask( pxSMBusProfile, SMBUS_REG_TGT_CONTROL_7, SMBUS_TGT_CONTROL_7_ADDRESS_MASK, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Passes the ulValue to the desired TgtControl register dependent
*           on the instance value
*
*******************************************************************************/
void vSMBusHWWriteTgtControlAddress( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucInstance, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        switch( ucInstance )
        {
        case 0:
            prvvSMBusHWWriteTgtControl0Address( pxSMBusProfile, ulValue );
            break;
        
        case 1:
            prvvSMBusHWWriteTgtControl1Address( pxSMBusProfile, ulValue );
            break;
        
        case 2:
            prvvSMBusHWWriteTgtControl2Address( pxSMBusProfile, ulValue );
            break;
        
        case 3:
            prvvSMBusHWWriteTgtControl3Address( pxSMBusProfile, ulValue );
            break;
        
        case 4:
            prvvSMBusHWWriteTgtControl4Address( pxSMBusProfile, ulValue );
            break;
        
        case 5:
            prvvSMBusHWWriteTgtControl5Address( pxSMBusProfile, ulValue );
            break;
        
        case 6:
            prvvSMBusHWWriteTgtControl6Address( pxSMBusProfile, ulValue );
            break;
        
        case 7:
            prvvSMBusHWWriteTgtControl7Address( pxSMBusProfile, ulValue );
            break;
        
        default:
            break;
        }
    }
}

/*******************************************************************************
*
* @brief    Passes the ulValue to the desired TgtControl register dependent
*           on the instance value
*
*******************************************************************************/
void vSMBusHWWriteTgtControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint8_t ucInstance, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        switch( ucInstance )
        {
        case 0:
            prvvSMBusHWWriteTgtControl0Enable( pxSMBusProfile, ulValue );
            break;
        
        case 1:
            prvvSMBusHWWriteTgtControl1Enable( pxSMBusProfile, ulValue );
            break;
        
        case 2:
            prvvSMBusHWWriteTgtControl2Enable( pxSMBusProfile, ulValue );
            break;
        
        case 3:
            prvvSMBusHWWriteTgtControl3Enable( pxSMBusProfile, ulValue );
            break;
        
        case 4:
            prvvSMBusHWWriteTgtControl4Enable( pxSMBusProfile, ulValue );
            break;
        
        case 5:
            prvvSMBusHWWriteTgtControl5Enable( pxSMBusProfile, ulValue );
            break;
        
        case 6:
            prvvSMBusHWWriteTgtControl6Enable( pxSMBusProfile, ulValue );
            break;
        
        case 7:
            prvvSMBusHWWriteTgtControl7Enable( pxSMBusProfile, ulValue );
            break;
        
        default:
            break;
        }
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_DATA_HOLD */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_DATA_HOLD
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlDataHold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_DATA_HOLD, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_START_HOLD */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_START_HOLD
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlStartHold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_START_HOLD, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_START_SETUP */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_START_SETUP
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlStartSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_START_SETUP, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_STOP_SETUP */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_STOP_SETUP
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlStopSetup( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_STOP_SETUP, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CLK_TLOW */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CLK_TLOW
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlClkTLow( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CLK_TLOW, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CLK_THIGH */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CLK_THIGH
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlClkTHigh( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CLK_THIGH, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_TEXT_PRESCALER */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_TEXT_PRESCALER
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlTextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_TEXT_PRESCALER, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlTextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_TEXT_TIMEOUT, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_TEXT_MAX */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_TEXT_MAX
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlTextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_TEXT_MAX, ulValue );
    }
}

/******************************************************************************************************************/
/* SMBUS_REG_PHY_CTLR_CEXT_PRESCALER */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CEXT_PRESCALER
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlCextPrescaler( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CEXT_PRESCALER, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlCextTimeout( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CEXT_TIMEOUT, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_PHY_CTLR_CEXT_MAX */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_PHY_CTLR_CEXT_MAX
*
*******************************************************************************/
void vSMBusHWWritePHYCtrlCextMax( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_PHY_CTLR_CEXT_MAX, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_CONTROL */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_CTLR_CONTROL
*
*******************************************************************************/
void vSMBusHWWriteCtrlControlEnable( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_CTLR_CONTROL, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_DESC_FIFO */
/*******************************************************************************
*
* @brief    Writes ulValue to hardware register SMBUS_REG_CTLR_DESC_FIFO
*
*******************************************************************************/
void vSMBusHWWriteCtrlDescFifo( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_CTLR_DESC_FIFO, ulValue );
    }
}

/*******************************************************************************
*
* @brief    Writes ulValue to RESET bitfield of hardware register SMBUS_REG_CTLR_DESC_FIFO
*
*******************************************************************************/
void vSMBusHWWriteCtrlDescFifoReset( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_CTLR_DESC_FIFO_RESET_FIELD_POSITION;
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_CTLR_DESC_FIFO, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_RX_FIFO */
/*******************************************************************************
*
* @brief    Writes ulValue to RESET bitfield of hardware register
*           SMBUS_REG_CTLR_RX_FIFO
*
*******************************************************************************/
void vSMBusHWWriteCtrlRxFifoReset( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        ulValue = ulValue << SMBUS_CTLR_RX_FIFO_RESET_FIELD_POSITION;
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO, ulValue );
    }
}

/******************************************************************************************************************/

/* SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD */
/*******************************************************************************
*
* @brief    Writes ulValue to FILL_THRESHOLD bitfield of hardware register
*           SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD
*
*******************************************************************************/
void vSMBusHWWriteCtrlRxFifoFillThreshold( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulValue )
{
    if( NULL != pxSMBusProfile )
    {
        prvvSMBusHardwareWrite( pxSMBusProfile, SMBUS_REG_CTLR_RX_FIFO_FILL_THRESHOLD, 
                        ( SMBUS_CTLR_RX_FIFO_FILL_THRESHOLD_FILL_THRESHOLD_MASK & ulValue ) );
    }
}
